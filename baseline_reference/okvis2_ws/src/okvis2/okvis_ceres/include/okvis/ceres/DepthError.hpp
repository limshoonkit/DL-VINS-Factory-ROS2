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
 * @file DepthError.hpp
 * @brief Header file for the PoseError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_CERES_DEPTHERROR_HPP_
#define INCLUDE_OKVIS_CERES_DEPTHERROR_HPP_

#include <ceres/sized_cost_function.h>

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Absolute error of a pose.
template<bool ONESIDED>
class DepthErrorT : public ::ceres::SizedCostFunction<
                     1 /* number of residuals */,
                     7 /* size of first parameter */,
                     4 /* size of second parameter */,
                     7 /* size of third parameter (camera extrinsics) */>, public ErrorInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<1, 7, 4, 7> base_t;

  static const bool onesided = ONESIDED; ///< If true, larger depth will not be penalised.

  /// \brief The number of residuals (1).
  static const int kNumResiduals = 1;

  /// \brief Default constructor.
  DepthErrorT() {}

  /// \brief Construct with measurement and information.
  /// @param[in] measurement The measurement.
  /// @param[in] stdev The depth standard deviation.
  DepthErrorT(double measurement, double stdev)
  {
    measurement_ = measurement;
    information_ = 1.0/(stdev*stdev);
    squareRootInformation_ = sqrt(information_);
  }

  /// \brief Trivial destructor.
  virtual ~DepthErrorT() override = default;

  /// \name Setters
  /// \{

  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  void setMeasurement(double measurement) { measurement_ = measurement; }

  /// \brief Set the information.
  /// @param[in] stdev The depth standard deviation.
  void setStdev(double stdev)
  {
    information_ = 1.0/(stdev*stdev);
    squareRootInformation_ = sqrt(information_);
  }

  /// \}
  /// \name Getters
  /// \{

  /// \brief Get the measurement.
  /// \return The measurement vector.
  double measurement() const { return measurement_; }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  double information() const { return information_; }

  /// \}

  // error term and Jacobian implementation
  /**
    * @brief This evaluates the error term and additionally computes the Jacobians.
    * @param parameters Pointer to the parameters (see ceres)
    * @param residuals Pointer to the residual vector (see ceres)
    * @param jacobians Pointer to the Jacobians (see ceres)
    * @return success of th evaluation.
    */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const override final {
    return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
  }

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const *const *parameters,
                                            double *residuals,
                                            double **jacobians,
                                            double **jacobiansMinimal) const override final
  {
    // We avoid the use of kinematics::Transformation here due to quaternion normalization etc.
    // This only matters in order to be able to check Jacobians with num. differentiation chained,
    // first w.r.t. q and then d_alpha.

    // pose: world to sensor transformation
    Eigen::Map<const Eigen::Vector3d> t_WS_W(&parameters[0][0]);
    Eigen::Quaterniond q_WS(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    q_WS.normalize();

    // the point in world coordinates
    Eigen::Map<const Eigen::Vector4d> hp_W(&parameters[1][0]);

    // the sensor to camera transformation
    Eigen::Map<const Eigen::Vector3d> t_SC_S(&parameters[2][0]);
    Eigen::Quaterniond q_SC(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    q_SC.normalize();

    // transform the point into the camera:
    Eigen::Matrix3d C_SC = q_SC.toRotationMatrix();
    Eigen::Matrix3d C_CS = C_SC.transpose();
    Eigen::Matrix4d T_CS = Eigen::Matrix4d::Identity();
    T_CS.topLeftCorner<3, 3>() = C_CS;
    T_CS.topRightCorner<3, 1>() = -C_CS * t_SC_S;
    Eigen::Matrix3d C_WS = q_WS.toRotationMatrix();
    Eigen::Matrix3d C_SW = C_WS.transpose();
    Eigen::Matrix4d T_SW = Eigen::Matrix4d::Identity();
    T_SW.topLeftCorner<3, 3>() = C_SW;
    T_SW.topRightCorner<3, 1>() = -C_SW * t_WS_W;
    Eigen::Vector4d hp_S = T_SW * hp_W;
    Eigen::Vector4d hp_C = T_CS * hp_S;
    double p_z = 0.0;

    // check if this can be ignored
    bool ignore = false;
    if (fabs(hp_C[3]) < 1.0e-16) {
      ignore = true;
    } else if (onesided) {
      p_z = hp_C[2] / hp_C[3];
      if (p_z > measurement_) {
        ignore = true;
      }
    }

    // compute error
    const double error = measurement_ - p_z;

    // weighted error:
    if (ignore) {
      residuals[0] = 0.0;
    } else {
      residuals[0] = squareRootInformation_ * error;
    }

    // calculate jacobians, if required
    if (jacobians != nullptr) {
      Eigen::Matrix<double, 1, 4> Jh_weighted(0.0, 0.0, squareRootInformation_ / hp_C[3], 0.0);
      if (jacobians[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J0(jacobians[0]);
        if (ignore) {
          J0.setZero();
          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[0] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J0_minimal_mapped(
                jacobiansMinimal[0]);
              J0_minimal_mapped.setZero();
            }
          }
        } else {
          Eigen::Vector3d p = hp_W.head<3>() - t_WS_W * hp_W[3];
          Eigen::Matrix<double, 4, 6> J;
          J.setZero();
          J.topLeftCorner<3, 3>() = C_SW * hp_W[3];
          J.topRightCorner<3, 3>() = -C_SW * okvis::kinematics::crossMx(p);

          // compute the minimal version
          Eigen::Matrix<double, 1, 6, Eigen::RowMajor> J0_minimal;
          J0_minimal = Jh_weighted * T_CS * J;


          // pseudo inverse of the local parametrization Jacobian:
          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
          PoseManifold::minusJacobian(parameters[0], J_lift.data());

          // hallucinate Jacobian w.r.t. state
          J0 = J0_minimal * J_lift;

          // if requested, provide minimal Jacobians
          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[0] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J0_minimal_mapped(
                jacobiansMinimal[0]);
              J0_minimal_mapped = J0_minimal;
            }
          }
        }
      }
      if (jacobians[1] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor> > J1(
          jacobians[1]); // map the raw pointer to an Eigen matrix for convenience
        if (ignore) {
          J1.setZero();
          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[1] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J1_minimal_mapped(
                jacobiansMinimal[1]);
              J1_minimal_mapped.setZero();
            }
          }
        } else {
          Eigen::Matrix4d T_CW = (T_CS * T_SW);
          J1 = -Jh_weighted * T_CW;

          // if requested, provide minimal Jacobians
          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[1] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J1_minimal_mapped(
                jacobiansMinimal[1]);
              Eigen::Matrix<double, 4, 3> S;
              S.setZero();
              S.topLeftCorner<3, 3>().setIdentity();
              J1_minimal_mapped = J1 * S; // this is for Euclidean-style perturbation only.
            }
          }
        }
      }
      if (jacobians[2] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J2(jacobians[2]);
        if (ignore) {
          J2.setZero();
          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[2] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J2_minimal_mapped(
                jacobiansMinimal[2]);
              J2_minimal_mapped.setZero();
            }
          }
        } else {
          Eigen::Vector3d p = hp_S.head<3>() - t_SC_S * hp_S[3];
          Eigen::Matrix<double, 4, 6> J;
          J.setZero();
          J.topLeftCorner<3, 3>() = C_CS * hp_S[3];
          J.topRightCorner<3, 3>() = -C_CS * okvis::kinematics::crossMx(p);

          // compute the minimal version
          Eigen::Matrix<double, 1, 6, Eigen::RowMajor> J2_minimal;
          J2_minimal = Jh_weighted * J;

          // pseudo inverse of the local parametrization Jacobian:
          Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
          PoseManifold::minusJacobian(parameters[2], J_lift.data());

          // hallucinate Jacobian w.r.t. state
          J2 = J2_minimal * J_lift;

          // if requested, provide minimal Jacobians
          if (jacobiansMinimal != nullptr) {
            if (jacobiansMinimal[2] != nullptr) {
              Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > J2_minimal_mapped(
                jacobiansMinimal[2]);
              J2_minimal_mapped = J2_minimal;
            }
          }
        }
      }
    }

    return true;

  }

  // sizes
  /// \brief Residual dimension.
  int residualDim() const override final {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  int parameterBlocks() const override final {
    return int(base_t::parameter_block_sizes().size());
  }

  /// \brief Dimension of an individual parameter block.
  int parameterBlockDim(int parameterBlockId) const override final{
    return base_t::parameter_block_sizes().at(size_t(parameterBlockId));
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const override final {
    return "DepthError";
  }

 protected:

  // the measurement
  double measurement_; ///< The pose measurement.

  // weighting related
  double information_; ///< The information.
  double squareRootInformation_; ///< The square root information.

};

/// \brief Template spectialisation of depth error term that is one-sided.
typedef DepthErrorT<true> OneSidedDepthError;

/// \brief Template specialisation for regular depth error term.
typedef DepthErrorT<false> DepthError;

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_DEPTHERROR_HPP_ */
