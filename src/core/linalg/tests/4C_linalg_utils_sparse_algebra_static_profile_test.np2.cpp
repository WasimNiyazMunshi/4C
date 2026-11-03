// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_utils_sparse_algebra_static_profile_test.hpp"

#include "4C_coupling_adapter_converter.hpp"

#include <array>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  // Only the row-map methods are used by MatrixLogicalSplitAndTransform in this test.
  class IdentityRowConverter : public Coupling::Adapter::CouplingConverter
  {
   public:
    IdentityRowConverter(std::shared_ptr<const Core::LinAlg::Map> source,
        std::shared_ptr<const Core::LinAlg::Map> destination)
        : source_(std::move(source)), destination_(std::move(destination))
    {
    }

    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> src_to_dst(
        [[maybe_unused]] std::shared_ptr<const Core::LinAlg::Vector<double>> source) const override
    {
      return nullptr;
    }

    [[nodiscard]] std::shared_ptr<Core::LinAlg::Vector<double>> dst_to_src(
        [[maybe_unused]] std::shared_ptr<const Core::LinAlg::Vector<double>> destination)
        const override
    {
      return nullptr;
    }

    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> src_map() const override
    {
      return source_;
    }
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> dst_map() const override
    {
      return destination_;
    }
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> perm_src_map() const override
    {
      return destination_;
    }
    [[nodiscard]] std::shared_ptr<const Core::LinAlg::Map> perm_dst_map() const override
    {
      return source_;
    }

    void fill_src_to_dst_map(std::map<int, int>& rows) const override
    {
      for (int i = 0; i < source_->num_my_elements(); ++i) rows[source_->gid(i)] = source_->gid(i);
    }

   private:
    std::shared_ptr<const Core::LinAlg::Map> source_;
    std::shared_ptr<const Core::LinAlg::Map> destination_;
  };

  // Rank one starts with no rows, but must receive a row wider than its local row maximum.
  TEST_F(SparseAlgebraStaticProfileTest, ImportRowsOntoRankWithNoSourceRows)
  {
    const int rank = Core::Communication::my_mpi_rank(comm_);
    const std::array source_rows = {0, 1};
    auto source_map = std::make_shared<Core::LinAlg::Map>(
        -1, rank == 0 ? 2 : 0, rank == 0 ? source_rows.data() : nullptr, 0, comm_);
    const int destination_row = rank;
    auto destination_map = std::make_shared<Core::LinAlg::Map>(-1, 1, &destination_row, 0, comm_);
    const std::array columns = {10, 11, 12};
    Core::LinAlg::Map domain_map(
        -1, rank == 0 ? 3 : 0, rank == 0 ? columns.data() : nullptr, 0, comm_);

    Core::LinAlg::SparseMatrix source(*source_map, 3, StorageProfile::static_profile);
    if (rank == 0)
    {
      source.assemble(7.0, 0, 10);
      for (int i = 0; i < 3; ++i) source.assemble(1.0 + i, 1, columns[i]);
    }
    source.complete(domain_map, *source_map);
    EXPECT_EQ(source.max_num_entries(), rank == 0 ? 3 : 0);
    EXPECT_EQ(source.global_max_num_entries(), 3);

    auto redistributed = Core::LinAlg::redistribute(source, *destination_map, domain_map);
    Core::LinAlg::SparseMatrix transformed(*destination_map, 3);
    IdentityRowConverter converter(source_map, destination_map);
    Coupling::Adapter::MatrixRowTransform transform;
    ASSERT_TRUE(transform(source, 1.0, converter, transformed));
    transformed.complete(domain_map, *destination_map);

    for (const Core::LinAlg::SparseMatrix* result : {redistributed.get(), &transformed})
    {
      EXPECT_TRUE(result->row_map().same_as(*destination_map));
      EXPECT_TRUE(result->domain_map().same_as(domain_map));
      int count = 0;
      std::array<double, 3> values{};
      std::array<int, 3> indices{};
      result->extract_global_row_copy(destination_row, 3, count, values.data(), indices.data());
      ASSERT_EQ(count, rank == 0 ? 1 : 3);
      for (int i = 0; i < count; ++i)
      {
        EXPECT_GE(indices[i], 10);
        EXPECT_LE(indices[i], 12);
        EXPECT_DOUBLE_EQ(values[i], rank == 0 ? 7.0 : indices[i] - 9.0);
      }
    }
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE
