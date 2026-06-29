/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>

namespace uosm
{
    namespace perception
    {

        namespace marginalization_utils
        {

            // Convert global parameter size to local (tangent space) size
            inline int globalToLocalSize(int size) noexcept
            {
                return size == 7 ? 6 : size;
            }

            // Compute quaternion difference in tangent space (axis-angle)
            inline Eigen::Vector3d quaternionDifference(const Eigen::Quaterniond &q,
                                                        const Eigen::Quaterniond &q0)
            {
                Eigen::Quaterniond dq = q0.inverse() * q;

                if (dq.w() < 0.0)
                    dq.coeffs() = -dq.coeffs();

                const double w = dq.w();
                const Eigen::Vector3d &v = dq.vec();

                if (w >= 1.0 - 1e-12)
                    return 2.0 * v;

                const double half_angle = std::acos(std::clamp(w, -1.0, 1.0));
                const double sin_half = std::sin(half_angle);

                if (std::abs(sin_half) < 1e-12)
                    return 2.0 * v;

                return (2.0 * half_angle / sin_half) * v;
            }

        } // namespace marginalization_utils

        // Stores a cost function and evaluation results for marginalization
        class ResidualBlockInfo
        {
        public:
            ResidualBlockInfo(std::unique_ptr<ceres::CostFunction> cost_function,
                              ceres::LossFunction *loss_function,
                              std::vector<double *> parameter_blocks,
                              std::vector<int> drop_set)
                : cost_function_(std::move(cost_function)),
                  loss_function_(loss_function),
                  parameter_blocks_(std::move(parameter_blocks)),
                  drop_set_(std::move(drop_set)) {}

            ResidualBlockInfo(ceres::CostFunction *cost_function,
                              ceres::LossFunction *loss_function,
                              std::vector<double *> parameter_blocks,
                              std::vector<int> drop_set)
                : cost_function_(cost_function),
                  loss_function_(loss_function),
                  parameter_blocks_(std::move(parameter_blocks)),
                  drop_set_(std::move(drop_set)) {}

            bool Evaluate();

            const ceres::CostFunction *costFunction() const { return cost_function_.get(); }
            const std::vector<double *> &parameterBlocks() const { return parameter_blocks_; }
            const std::vector<int> &dropSet() const { return drop_set_; }
            const std::vector<Eigen::MatrixXd> &jacobians() const { return jacobians_; }
            const Eigen::VectorXd &residuals() const { return residuals_; }

        private:
            std::unique_ptr<ceres::CostFunction> cost_function_;
            ceres::LossFunction *loss_function_;
            std::vector<double *> parameter_blocks_;
            std::vector<int> drop_set_;
            std::vector<Eigen::MatrixXd> jacobians_;
            Eigen::VectorXd residuals_;

            friend class MarginalizationInfo;
        };

        // Manages marginalization via Schur complement on normal equations
        class MarginalizationInfo
        {
        public:
            static constexpr double kEpsilon = 1e-8;

            MarginalizationInfo() = default;
            ~MarginalizationInfo() = default;

            MarginalizationInfo(const MarginalizationInfo &) = delete;
            MarginalizationInfo &operator=(const MarginalizationInfo &) = delete;
            MarginalizationInfo(MarginalizationInfo &&) = default;
            MarginalizationInfo &operator=(MarginalizationInfo &&) = default;

            void addResidualBlockInfo(std::unique_ptr<ResidualBlockInfo> residual_block_info);

            void addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
            {
                addResidualBlockInfo(std::unique_ptr<ResidualBlockInfo>(residual_block_info));
            }

            void preMarginalize();
            void marginalize(int max_threads = 0);

            std::vector<double *> getParameterBlocks(std::unordered_map<std::uintptr_t, double *> &addr_shift);

            bool isValid() const { return valid_; }
            int remainingSize() const { return n_; }
            const std::vector<int> &keepBlockSize() const { return keep_block_size_; }
            const std::vector<int> &keepBlockIdx() const { return keep_block_idx_; }
            const std::vector<double *> &keepBlockData() const { return keep_block_data_; }
            const Eigen::VectorXd &linearizedResiduals() const { return linearized_residuals_; }
            const Eigen::MatrixXd &linearizedJacobians() const { return linearized_jacobians_; }

            bool valid = false;
            int m = 0, n = 0;
            std::vector<int> keep_block_size;
            std::vector<int> keep_block_idx;
            std::vector<double *> keep_block_data;
            Eigen::VectorXd linearized_residuals;
            Eigen::MatrixXd linearized_jacobians;

        private:
            bool marginalizeSchurComplement(int total_size, int max_threads);
            void syncLegacyMembers();

            bool valid_ = false;
            int m_ = 0;
            int n_ = 0;

            std::unordered_map<std::uintptr_t, int> parameter_block_size_;
            std::unordered_map<std::uintptr_t, int> parameter_block_idx_;
            std::unordered_map<std::uintptr_t, std::unique_ptr<double[]>> parameter_block_data_;
            std::vector<std::unique_ptr<ResidualBlockInfo>> factors_;

            std::vector<int> keep_block_size_;
            std::vector<int> keep_block_idx_;
            std::vector<double *> keep_block_data_;
            Eigen::VectorXd linearized_residuals_;
            Eigen::MatrixXd linearized_jacobians_;
        };

        // Ceres cost function wrapping marginalization prior
        class MarginalizationFactor : public ceres::CostFunction
        {
        public:
            explicit MarginalizationFactor(MarginalizationInfo *marginalization_info);

            bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const override;

            MarginalizationInfo *marginalization_info;

        private:
            MarginalizationInfo *marginalization_info_;
        };

        inline bool ResidualBlockInfo::Evaluate()
        {
            if (!cost_function_)
                return false;

            const int num_residuals = cost_function_->num_residuals();
            const std::vector<int> &block_sizes = cost_function_->parameter_block_sizes();

            if (block_sizes.size() != parameter_blocks_.size())
                return false;

            residuals_.resize(num_residuals);

            // Allocate global-size Jacobians
            std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                jacobians_global(block_sizes.size());
            std::vector<double *> jacobian_ptrs(block_sizes.size());

            for (size_t i = 0; i < block_sizes.size(); ++i)
            {
                jacobians_global[i].resize(num_residuals, block_sizes[i]);
                jacobians_global[i].setZero();
                jacobian_ptrs[i] = jacobians_global[i].data();
            }

            if (!cost_function_->Evaluate(parameter_blocks_.data(), residuals_.data(), jacobian_ptrs.data()))
                return false;

            // Extract local-size Jacobians
            jacobians_.resize(block_sizes.size());
            for (size_t i = 0; i < block_sizes.size(); ++i)
            {
                const int local_size = marginalization_utils::globalToLocalSize(block_sizes[i]);
                jacobians_[i] = jacobians_global[i].leftCols(local_size);
            }

            // Apply robust loss function
            if (loss_function_)
            {
                const double sq_norm = residuals_.squaredNorm();
                double rho[3];
                loss_function_->Evaluate(sq_norm, rho);

                const double sqrt_rho1 = std::sqrt(rho[1]);
                double residual_scaling, alpha_sq_norm;

                if (sq_norm == 0.0 || rho[2] <= 0.0)
                {
                    residual_scaling = sqrt_rho1;
                    alpha_sq_norm = 0.0;
                }
                else
                {
                    const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
                    const double alpha = 1.0 - std::sqrt(D);
                    residual_scaling = sqrt_rho1 / (1.0 - alpha);
                    alpha_sq_norm = alpha / sq_norm;
                }

                for (size_t i = 0; i < block_sizes.size(); ++i)
                {
                    jacobians_[i] = sqrt_rho1 * (jacobians_[i] -
                                                 alpha_sq_norm * residuals_ * (residuals_.transpose() * jacobians_[i]));
                }
                residuals_ *= residual_scaling;
            }

            return true;
        }

        inline void MarginalizationInfo::addResidualBlockInfo(
            std::unique_ptr<ResidualBlockInfo> residual_block_info)
        {
            if (!residual_block_info || !residual_block_info->costFunction())
                return;

            const auto &param_blocks = residual_block_info->parameterBlocks();
            const std::vector<int> &block_sizes =
                residual_block_info->costFunction()->parameter_block_sizes();

            for (size_t i = 0; i < param_blocks.size(); ++i)
            {
                const auto addr = reinterpret_cast<std::uintptr_t>(param_blocks[i]);
                parameter_block_size_[addr] = block_sizes[i];
            }

            for (int idx : residual_block_info->dropSet())
            {
                if (idx >= 0 && idx < static_cast<int>(param_blocks.size()))
                {
                    const auto addr = reinterpret_cast<std::uintptr_t>(param_blocks[idx]);
                    parameter_block_idx_[addr] = 0;
                }
            }

            factors_.push_back(std::move(residual_block_info));
        }

        inline void MarginalizationInfo::preMarginalize()
        {
            for (auto &factor : factors_)
            {
                if (!factor->Evaluate())
                    continue;

                const auto &param_blocks = factor->parameterBlocks();
                const std::vector<int> &block_sizes = factor->costFunction()->parameter_block_sizes();

                for (size_t i = 0; i < param_blocks.size(); ++i)
                {
                    const auto addr = reinterpret_cast<std::uintptr_t>(param_blocks[i]);
                    const int size = block_sizes[i];

                    if (parameter_block_data_.find(addr) == parameter_block_data_.end())
                    {
                        auto data = std::make_unique<double[]>(size);
                        std::memcpy(data.get(), param_blocks[i], sizeof(double) * size);
                        parameter_block_data_[addr] = std::move(data);
                    }
                }
            }
        }

        inline void MarginalizationInfo::marginalize(int max_threads)
        {
            using namespace marginalization_utils;

            if (factors_.empty())
            {
                valid_ = false;
                syncLegacyMembers();
                return;
            }

            int pos = 0;
            for (auto &it : parameter_block_idx_)
            {
                it.second = pos;
                pos += globalToLocalSize(parameter_block_size_[it.first]);
            }
            m_ = pos;

            if (m_ == 0)
            {
                valid_ = false;
                syncLegacyMembers();
                return;
            }

            for (const auto &it : parameter_block_size_)
            {
                if (parameter_block_idx_.find(it.first) == parameter_block_idx_.end())
                {
                    parameter_block_idx_[it.first] = pos;
                    pos += globalToLocalSize(it.second);
                }
            }
            n_ = pos - m_;

            bool success = marginalizeSchurComplement(pos, max_threads);

            if (success)
            {
                valid_ = (n_ > 0);
            }
            else
            {
                valid_ = false;
            }

            syncLegacyMembers();
        }

        inline bool MarginalizationInfo::marginalizeSchurComplement(int total_size, int max_threads)
        {
            if (n_ <= 0)
                return false;

            const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
            const int thread_cap = (max_threads > 0) ? max_threads : std::max(1, hw_threads - 2);
            const int num_threads = std::min(thread_cap,
                                             std::max(1, static_cast<int>(factors_.size())));

            struct ThreadData
            {
                Eigen::MatrixXd A;
                Eigen::VectorXd b;
                std::vector<ResidualBlockInfo *> sub_factors;
            };

            std::vector<ThreadData> thread_data(num_threads);
            for (int t = 0; t < num_threads; ++t)
            {
                thread_data[t].A = Eigen::MatrixXd::Zero(total_size, total_size);
                thread_data[t].b = Eigen::VectorXd::Zero(total_size);
            }

            // Distribute factors round-robin
            for (size_t f = 0; f < factors_.size(); ++f)
                thread_data[f % num_threads].sub_factors.push_back(factors_[f].get());

            // Worker function: accumulate J^TJ into thread-local A, b
            auto worker = [this](ThreadData &td)
            {
                for (auto *factor : td.sub_factors)
                {
                    const std::vector<int> &block_sizes = factor->costFunction()->parameter_block_sizes();
                    const auto &param_blocks = factor->parameterBlocks();

                    for (size_t i = 0; i < param_blocks.size(); ++i)
                    {
                        int idx_i = parameter_block_idx_.at(reinterpret_cast<std::uintptr_t>(param_blocks[i]));
                        int size_i = marginalization_utils::globalToLocalSize(block_sizes[i]);
                        const Eigen::MatrixXd &jacobian_i = factor->jacobians()[i];

                        for (size_t j = i; j < param_blocks.size(); ++j)
                        {
                            int idx_j = parameter_block_idx_.at(reinterpret_cast<std::uintptr_t>(param_blocks[j]));
                            int size_j = marginalization_utils::globalToLocalSize(block_sizes[j]);
                            const Eigen::MatrixXd &jacobian_j = factor->jacobians()[j];

                            if (i == j)
                            {
                                td.A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                            }
                            else
                            {
                                td.A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                                td.A.block(idx_j, idx_i, size_j, size_i) += jacobian_j.transpose() * jacobian_i;
                            }
                        }
                        td.b.segment(idx_i, size_i) += jacobian_i.transpose() * factor->residuals();
                    }
                }
            };

            // Launch threads
            std::vector<std::thread> threads;
            for (int t = 1; t < num_threads; ++t)
                threads.emplace_back(worker, std::ref(thread_data[t]));
            worker(thread_data[0]); // Use current thread for first batch

            // Collect results
            Eigen::MatrixXd A = std::move(thread_data[0].A);
            Eigen::VectorXd b = std::move(thread_data[0].b);
            for (auto &th : threads)
                th.join();
            for (int t = 1; t < num_threads; ++t)
            {
                A += thread_data[t].A;
                b += thread_data[t].b;
            }

            // Symmetrize Amm for numerical stability
            Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m_, m_) + A.block(0, 0, m_, m_).transpose());

            // Invert Amm via eigendecomposition (handles rank deficiency)
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
            Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > kEpsilon).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();

            // Schur complement: A_sc = Arr - Arm * Amm^-1 * Amr
            Eigen::VectorXd bmm = b.segment(0, m_);
            Eigen::MatrixXd Amr = A.block(0, m_, m_, n_);
            Eigen::MatrixXd Arm = A.block(m_, 0, n_, m_);
            Eigen::MatrixXd Arr = A.block(m_, m_, n_, n_);
            Eigen::VectorXd brr = b.segment(m_, n_);

            A = Arr - Arm * Amm_inv * Amr;
            b = brr - Arm * Amm_inv * bmm;

            // Eigendecompose the Schur complement for sqrt factorization
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
            Eigen::VectorXd S = Eigen::VectorXd(
                (saes2.eigenvalues().array() > kEpsilon).select(saes2.eigenvalues().array(), 0));
            Eigen::VectorXd S_inv = Eigen::VectorXd(
                (saes2.eigenvalues().array() > kEpsilon).select(saes2.eigenvalues().array().inverse(), 0));

            linearized_jacobians_ = S.cwiseSqrt().asDiagonal() * saes2.eigenvectors().transpose();
            linearized_residuals_ = S_inv.cwiseSqrt().asDiagonal() * saes2.eigenvectors().transpose() * b;

            return true;
        }

        inline void MarginalizationInfo::syncLegacyMembers()
        {
            valid = valid_;
            m = m_;
            n = n_;
            keep_block_size = keep_block_size_;
            keep_block_idx = keep_block_idx_;
            keep_block_data = keep_block_data_;
            linearized_residuals = linearized_residuals_;
            linearized_jacobians = linearized_jacobians_;
        }

        inline std::vector<double *> MarginalizationInfo::getParameterBlocks(
            std::unordered_map<std::uintptr_t, double *> &addr_shift)
        {
            std::vector<double *> keep_block_addr;
            keep_block_size_.clear();
            keep_block_idx_.clear();
            keep_block_data_.clear();

            for (const auto &it : parameter_block_idx_)
            {
                if (it.second >= m_)
                {
                    keep_block_size_.push_back(parameter_block_size_[it.first]);
                    keep_block_idx_.push_back(it.second - m_);
                    keep_block_data_.push_back(parameter_block_data_[it.first].get());
                    keep_block_addr.push_back(addr_shift[it.first]);
                }
            }

            syncLegacyMembers();
            return keep_block_addr;
        }

        inline MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *_marginalization_info)
            : marginalization_info(_marginalization_info),
              marginalization_info_(_marginalization_info)
        {
            if (!marginalization_info_)
                throw std::invalid_argument("MarginalizationFactor: null marginalization_info");

            for (int size : marginalization_info_->keepBlockSize())
                mutable_parameter_block_sizes()->push_back(size);

            set_num_residuals(marginalization_info_->remainingSize());
        }

        inline bool MarginalizationFactor::Evaluate(
            double const *const *parameters,
            double *residuals,
            double **jacobians) const
        {
            const int n = marginalization_info_->remainingSize();
            const auto &keep_block_size = marginalization_info_->keepBlockSize();
            const auto &keep_block_idx = marginalization_info_->keepBlockIdx();
            const auto &keep_block_data = marginalization_info_->keepBlockData();
            const auto &J = marginalization_info_->linearizedJacobians();
            const auto &r0 = marginalization_info_->linearizedResiduals();

            Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);

            for (size_t i = 0; i < keep_block_size.size(); ++i)
            {
                const int size = keep_block_size[i];
                const int local_size = marginalization_utils::globalToLocalSize(size);
                const int idx = keep_block_idx[i];

                Eigen::Map<const Eigen::VectorXd> x(parameters[i], size);
                Eigen::Map<const Eigen::VectorXd> x0(keep_block_data[i], size);

                if (size == 7)
                {
                    dx.segment<3>(idx) = x.head<3>() - x0.head<3>();
                    Eigen::Quaterniond q(x(6), x(3), x(4), x(5));
                    Eigen::Quaterniond q0(x0(6), x0(3), x0(4), x0(5));
                    dx.segment<3>(idx + 3) = marginalization_utils::quaternionDifference(q, q0);
                }
                else
                {
                    dx.segment(idx, local_size) = x - x0;
                }
            }

            Eigen::Map<Eigen::VectorXd>(residuals, n) = r0 + J * dx;

            if (jacobians)
            {
                for (size_t i = 0; i < keep_block_size.size(); ++i)
                {
                    if (jacobians[i])
                    {
                        const int size = keep_block_size[i];
                        const int local_size = marginalization_utils::globalToLocalSize(size);
                        const int idx = keep_block_idx[i];

                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                            jacobian(jacobians[i], n, size);

                        jacobian.setZero();
                        jacobian.leftCols(local_size) = J.middleCols(idx, local_size);
                    }
                }
            }

            return true;
        }

    } // namespace perception
} // namespace uosm
