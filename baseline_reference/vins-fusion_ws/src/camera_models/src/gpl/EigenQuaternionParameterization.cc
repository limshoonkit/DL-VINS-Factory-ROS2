#include "camodocal/gpl/EigenQuaternionParameterization.h"

#include <cmath>

namespace camodocal
{

bool
EigenQuaternionParameterization::Plus(const double* x,
                                      const double* delta,
                                      double* x_plus_delta) const
{
    const double norm_delta =
        sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    if (norm_delta > 0.0)
    {
        const double sin_delta_by_delta = (sin(norm_delta) / norm_delta);
        double q_delta[4];
        q_delta[0] = sin_delta_by_delta * delta[0];
        q_delta[1] = sin_delta_by_delta * delta[1];
        q_delta[2] = sin_delta_by_delta * delta[2];
        q_delta[3] = cos(norm_delta);
        EigenQuaternionProduct(q_delta, x, x_plus_delta);
    }
    else
    {
        for (int i = 0; i < 4; ++i)
        {
            x_plus_delta[i] = x[i];
        }
    }
    return true;
}

bool
EigenQuaternionParameterization::PlusJacobian(const double* x,
                                              double* jacobian) const
{
    jacobian[0] =  x[3]; jacobian[1]  =  x[2]; jacobian[2]  = -x[1];  // NOLINT
    jacobian[3] = -x[2]; jacobian[4]  =  x[3]; jacobian[5]  =  x[0];  // NOLINT
    jacobian[6] =  x[1]; jacobian[7] = -x[0]; jacobian[8] =  x[3];  // NOLINT
    jacobian[9] = -x[0]; jacobian[10]  = -x[1]; jacobian[11]  = -x[2];  // NOLINT
    return true;
}

bool
EigenQuaternionParameterization::Minus(const double* y,
                                       const double* x,
                                       double* y_minus_x) const
{
    // Compute x_inverse * y
    // x_inverse for unit quaternion: conjugate
    double x_inv[4] = {-x[0], -x[1], -x[2], x[3]};
    double delta_q[4];
    EigenQuaternionProduct(x_inv, y, delta_q);

    // Convert quaternion to angle-axis
    // delta_q = [sin(theta/2) * axis, cos(theta/2)]
    const double sin_half_angle = sqrt(delta_q[0] * delta_q[0] +
                                       delta_q[1] * delta_q[1] +
                                       delta_q[2] * delta_q[2]);
    if (sin_half_angle > 0.0)
    {
        const double cos_half_angle = delta_q[3];
        const double two_theta = 2.0 * atan2(sin_half_angle, cos_half_angle);
        const double k = two_theta / sin_half_angle;
        y_minus_x[0] = k * delta_q[0];
        y_minus_x[1] = k * delta_q[1];
        y_minus_x[2] = k * delta_q[2];
    }
    else
    {
        y_minus_x[0] = 2.0 * delta_q[0];
        y_minus_x[1] = 2.0 * delta_q[1];
        y_minus_x[2] = 2.0 * delta_q[2];
    }
    return true;
}

bool
EigenQuaternionParameterization::MinusJacobian(const double* x,
                                               double* jacobian) const
{
    // The MinusJacobian at identity is the pseudo-inverse of PlusJacobian
    // For unit quaternion manifold, this is approximately:
    jacobian[0] =  x[3]; jacobian[1]  = -x[2]; jacobian[2]  =  x[1]; jacobian[3]  = -x[0];  // NOLINT
    jacobian[4] =  x[2]; jacobian[5]  =  x[3]; jacobian[6]  = -x[0]; jacobian[7]  = -x[1];  // NOLINT
    jacobian[8] = -x[1]; jacobian[9]  =  x[0]; jacobian[10] =  x[3]; jacobian[11] = -x[2];  // NOLINT
    return true;
}

}
