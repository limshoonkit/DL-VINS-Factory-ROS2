#ifndef EIGENQUATERNIONPARAMETERIZATION_H
#define EIGENQUATERNIONPARAMETERIZATION_H

#include <ceres/ceres.h>

namespace camodocal
{

class EigenQuaternionParameterization : public ceres::Manifold
{
public:
    virtual ~EigenQuaternionParameterization() {}
    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const override;
    bool PlusJacobian(const double* x,
                      double* jacobian) const override;
    bool Minus(const double* y,
               const double* x,
               double* y_minus_x) const override;
    bool MinusJacobian(const double* x,
                       double* jacobian) const override;
    int AmbientSize() const override { return 4; }
    int TangentSize() const override { return 3; }

private:
    template<typename T>
    void EigenQuaternionProduct(const T z[4], const T w[4], T zw[4]) const;
};


template<typename T>
void
EigenQuaternionParameterization::EigenQuaternionProduct(const T z[4], const T w[4], T zw[4]) const
{
    zw[0] = z[3] * w[0] + z[0] * w[3] + z[1] * w[2] - z[2] * w[1];
    zw[1] = z[3] * w[1] - z[0] * w[2] + z[1] * w[3] + z[2] * w[0];
    zw[2] = z[3] * w[2] + z[0] * w[1] - z[1] * w[0] + z[2] * w[3];
    zw[3] = z[3] * w[3] - z[0] * w[0] - z[1] * w[1] - z[2] * w[2];
}

}

#endif
