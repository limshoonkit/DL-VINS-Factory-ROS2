/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 *  Copyright (c) 2024, Smart Robotics Lab / Technical University of Munich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab, ETH Zurich, Smart Robotics Lab,
 *     Imperial College London, Technical University of Munich, nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/

/**
 * @file TwoPoseExtrinsicsGraphError.hpp
 * @brief Header file for the TwoPoseExtrinsicsGraphError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_TWOPOSEEXTRINSICSGRAPHERROR_HPP_
#define INCLUDE_OKVIS_CERES_TWOPOSEEXTRINSICSGRAPHERROR_HPP_

#include <vector>

#include <ceres/sized_cost_function.h>
#include <ceres/loss_function.h>

#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/ReprojectionErrorBase.hpp>
#include <okvis/ceres/TwoPoseGraphError.hpp>
#include <okvis/FrameTypedefs.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

class TwoPoseExtrinsicsGraphErrorConst;

/// \brief Relative error between two poses. \todo also use non-const extrinsics...
class TwoPoseExtrinsicsGraphError : public ::ceres::CostFunction
    // 6+N*6 number of residuals,
    // 7 size of first parameter (pose)
    // 7 size of second parameter (pose)
    // 7 size of third parameter (extrinsics)
    // ...
    , public TwoPoseGraphError {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Constructor.
  TwoPoseExtrinsicsGraphError() = delete;

  /// \brief Simple constructor with the two pose IDs.
  /// \param referencePoseId The reference pose ID.
  /// \param otherPoseId The ID of the other pose.
  /// \param numCams The number of cameras on the rig.
  /// \param stayConst If true (default), the reference and other state extrinsics are the same.
  TwoPoseExtrinsicsGraphError(StateId referencePoseId, StateId otherPoseId, size_t numCams,
                              bool stayConst = true);

  /// \brief Trivial destructor.
  virtual ~TwoPoseExtrinsicsGraphError() override = default;

  /// @brief Convert this relative pose error back into observations.
  /// @param[out] observations The observations originally passed.
  /// @param[out] duplicates Indicates which ones had been added originally as duplications.
  /// @return True on success.
  virtual bool convertToReprojectionErrors(
    std::vector<Observation> &observations,
    std::vector<KeypointIdentifier> &duplicates) override final;

  /// @brief This computes a relative pose error from the observations.
  /// @return True on success.
  virtual bool compute() override final;

  // error term and Jacobian implementation (inherited pure virtuals from ::ceres::CostFunction)
  /**
    * @brief This evaluates the error term and additionally computes the Jacobians.
    * @param parameters Pointer to the parameters (see ceres)
    * @param residuals Pointer to the residual vector (see ceres)
    * @param jacobians Pointer to the Jacobians (see ceres)
    * @return success of th evaluation.
    */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  // sizes
  /// \brief Residual dimension.
  int residualDim() const override final {
    return num_residuals();
  }

  /// \brief Number of parameter blocks.
  int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  /// \brief Dimension of an individual parameter block.
  int parameterBlockDim(int parameterBlockId) const override final {
    return parameter_block_sizes().at(size_t(parameterBlockId));
  }

  /// \brief Get a cloned relative pose error, but as a constant one (i.e. not back-convertalble)
  virtual std::shared_ptr<TwoPoseGraphErrorConst> cloneTwoPoseGraphErrorConst() const override final;

  /// @brief Residual block type as string
  virtual std::string typeInfo() const override final{
    return "TwoPoseExtrinsicsGraphError";
  }

  /// @brief The covisibilities between the two poses.
  /// @return The number of co-observed landmarks.
  //int covisibility(std::map<uint64_t, std::map<uint64_t, size_t>> & coObservationCounts) const;

  /// @brief The strength of this link (regarding position uncertainty only) -- use for
  /// visualisations.
  /// @return The strength as inverse position standard deviation [1/m].
  virtual double strength() const override final;

protected:

  /// @name The internal storage of the linearised system.
  /// @{
  Eigen::MatrixXd H00_;  ///< lhs - Hessian dense (sparse/sparse-dense comp. on the fly)
  Eigen::VectorXd b0_;  ///<  rhs constant part dense

  /// We use the formulation e = DelatX_i + T_WS_bar * (T_WS0^(-1)*T_WS)^(-1).
  /// With information matrix H = J_^T*J_.
  Eigen::VectorXd DeltaX_; ///< nominal deviation from linearisation point
  Eigen::MatrixXd J_; ///< the decomposed inf. matrix (of the relative system at lin pt)

  /// \brief Adding residual blocks will invalidate this.
  /// Before optimizing, call updateErrorComputation()
  bool errorComputationValid_;
};

/// @brief Constant (non-convertible) version of the TwoPoseExtrinsicsGraphError (see above).
class TwoPoseExtrinsicsGraphErrorConst : public ::ceres::CostFunction,
                                         public TwoPoseGraphErrorConst {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Default constructor.
  TwoPoseExtrinsicsGraphErrorConst() = delete;
  /// \brief To construct from TwoPoseExtrinsicsGraphError.
  TwoPoseExtrinsicsGraphErrorConst(
    const Eigen::VectorXd &DeltaX,
    const Eigen::MatrixXd &J,
    const kinematics::Transformation &linearisationPoint_T_S0S1,
    const std::vector<std::shared_ptr<kinematics::Transformation>> &linearisationPoints_T_SC);
  /// \brief Trivial destructor.
  virtual ~TwoPoseExtrinsicsGraphErrorConst() override = default;
  // error term and Jacobian implementation (inherited pure virtuals from ::ceres::CostFunction)
  /**
    * @brief This evaluates the error term and additionally computes the Jacobians.
    * @param parameters Pointer to the parameters (see ceres)
    * @param residuals Pointer to the residual vector (see ceres)
    * @param jacobians Pointer to the Jacobians (see ceres)
    * @return success of th evaluation.
    */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const override final;

  // sizes
  /// \brief Residual dimension.
  int residualDim() const override final {
    return num_residuals();
  }

  /// \brief Number of parameter blocks.
  int parameterBlocks() const override final {
    return int(parameter_block_sizes().size());
  }

  /// \brief Dimension of an individual parameter block.
  int parameterBlockDim(int parameterBlockId) const override final {
    return parameter_block_sizes().at(size_t(parameterBlockId));
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const override final {
    return "TwoPoseExtrinsicsGraphErrorConst";
  }

protected:

  /// @name The internal storage of the linearised system.
  /// @{

  /// We use the formulation e = DelatX_i + T_WS_bar * (T_WS0^(-1)*T_WS)^(-1).
  /// With information matrix H = J_^T*J_.
  Eigen::VectorXd DeltaX_; ///< nominal deviation from linearisation point
  Eigen::MatrixXd J_; ///< the decomposed info matrix (of the relative system at lin pt)
  kinematics::Transformation linearisationPoint_T_S0S1_; ///< Relative poses
  std::vector<std::shared_ptr<kinematics::Transformation>> linearisationPoints_T_SC_; ///< Extrins.
};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_TWOPOSEEXTRINSICSGRAPHERROR_HPP_ */
