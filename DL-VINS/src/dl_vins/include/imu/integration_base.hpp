/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include "../utility/math_util.hpp"
#include <ceres/ceres.h>

using namespace Eigen;

class IntegrationBase
{
public:
    IntegrationBase() = delete;
    IntegrationBase(const Eigen::Vector3d &_acc_0,
                    const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba,
                    const Eigen::Vector3d &_linearized_bg,
                    const Eigen::Vector3d &_gravity_norm,
                    const double acc_n_sq,
                    const double acc_w_sq,
                    const double gyr_n_sq,
                    const double gyr_w_sq,
                    bool use_rk4 = false)
        : acc_0{_acc_0},
          gyr_0{_gyr_0},
          linearized_acc{_acc_0},
          linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba},
          linearized_bg{_linearized_bg},
          gravity{_gravity_norm},
          ACC_N_SQ{acc_n_sq},
          ACC_W_SQ{acc_w_sq},
          GYR_N_SQ{gyr_n_sq},
          GYR_W_SQ{gyr_w_sq},
          use_rk4{use_rk4},
          sum_dt{0.0},
          delta_p{Eigen::Vector3d::Zero()},
          delta_q{Eigen::Quaterniond::Identity()},
          delta_v{Eigen::Vector3d::Zero()}
    {
        jacobian.setIdentity();
        covariance.setZero();

        noise.setZero();
        noise.block<3, 3>(0, 0) = ACC_N_SQ * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = GYR_N_SQ * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = ACC_N_SQ * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = GYR_N_SQ * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = ACC_W_SQ * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = GYR_W_SQ * Eigen::Matrix3d::Identity();

        dt_buf.reserve(2000);
        acc_buf.reserve(2000);
        gyr_buf.reserve(2000);
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();

        const size_t n = dt_buf.size();
        for (size_t i = 0; i < n; ++i)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi,
                                          const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai,
                                          const Eigen::Vector3d &Bgi, const Eigen::Vector3d &Pj,
                                          const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
                                          const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) const
    {
        Eigen::Matrix<double, 15, 1> residuals;

        const Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        const Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);
        const Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);
        const Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        const Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        const Eigen::Vector3d dba = Bai - linearized_ba;
        const Eigen::Vector3d dbg = Bgi - linearized_bg;

        const Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        const Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        const Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        const Eigen::Quaterniond Qi_inv = Qi.inverse();
        const double half_g_t2 = 0.5 * sum_dt * sum_dt;
        const double g_t = sum_dt;

        residuals.block<3, 1>(O_P, 0) = Qi_inv * (half_g_t2 * gravity + Pj - Pi - sum_dt * Vi) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2.0 * (corrected_delta_q.inverse() * (Qi_inv * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi_inv * (g_t * gravity + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;

        return residuals;
    }

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;
    Eigen::Vector3d linearized_ba, linearized_bg;
    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    const Eigen::Vector3d gravity;

private:
    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;

        if (use_rk4)
        {
            rungeKutta4Integration();
        }
        else
        {
            midPointIntegration();
        }

        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;
    }

    void midPointIntegration()
    {
        const Vector3d un_acc_0 = delta_q * (acc_0 - linearized_ba);
        const Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - linearized_bg;
        const Quaterniond q_mid(1, un_gyr(0) * dt * 0.5, un_gyr(1) * dt * 0.5, un_gyr(2) * dt * 0.5);

        delta_q = delta_q * q_mid;
        const Vector3d un_acc_1 = delta_q * (acc_1 - linearized_ba);
        const Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

        delta_p += delta_v * dt + 0.5 * un_acc * dt * dt;
        delta_v += un_acc * dt;

        updateJacobian(un_gyr, un_acc_0, un_acc_1, q_mid);
    }

    void rungeKutta4Integration()
    {
        const Vector3d acc_unbiased_0 = acc_0 - linearized_ba;
        const Vector3d acc_unbiased_1 = acc_1 - linearized_ba;
        const Vector3d gyr_unbiased_0 = gyr_0 - linearized_bg;
        const Vector3d gyr_unbiased_1 = gyr_1 - linearized_bg;

        // RK4 for rotation
        const Vector3d k1_omega = gyr_unbiased_0;
        const Vector3d k2_omega = 0.5 * (gyr_unbiased_0 + gyr_unbiased_1);
        const Vector3d k3_omega = k2_omega;
        const Vector3d k4_omega = gyr_unbiased_1;
        const Vector3d omega_rk4 = (k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega) / 6.0;

        const Quaterniond dq(1, omega_rk4(0) * dt * 0.5, omega_rk4(1) * dt * 0.5, omega_rk4(2) * dt * 0.5);
        const Quaterniond q_mid = delta_q * Quaterniond(1, k2_omega(0) * dt * 0.25, k2_omega(1) * dt * 0.25, k2_omega(2) * dt * 0.25);

        delta_q = delta_q * dq;

        // RK4 for velocity and position
        const Vector3d k1_v = delta_q * acc_unbiased_0;
        const Vector3d k2_v = q_mid * (0.5 * (acc_unbiased_0 + acc_unbiased_1));
        const Vector3d k3_v = k2_v;
        const Vector3d k4_v = delta_q * acc_unbiased_1;
        const Vector3d acc_rk4 = (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;

        delta_p += delta_v * dt + 0.5 * acc_rk4 * dt * dt;
        delta_v += acc_rk4 * dt;

        // Approximate Jacobian update
        updateJacobian(omega_rk4, k1_v, k4_v, dq);
    }

    inline void updateJacobian(const Vector3d &un_gyr, const Vector3d &un_acc_0,
                               const Vector3d &un_acc_1, const Quaterniond &dq)
    {
        const Vector3d a_0_x = acc_0 - linearized_ba;
        const Vector3d a_1_x = acc_1 - linearized_ba;

        const Matrix3d R_w_x = skewSymmetric(un_gyr);
        const Matrix3d R_a_0_x = skewSymmetric(a_0_x);
        const Matrix3d R_a_1_x = skewSymmetric(a_1_x);

        const Matrix3d delta_R = delta_q.toRotationMatrix();
        const Matrix3d result_R = (delta_q * dq).toRotationMatrix();

        Eigen::Matrix<double, 15, 15> F;
        F.setZero();

        F.block<3, 3>(0, 0) = Matrix3d::Identity();
        F.block<3, 3>(0, 3) = -0.25 * (delta_R * R_a_0_x + result_R * R_a_1_x * (Matrix3d::Identity() - R_w_x * dt)) * dt * dt;
        F.block<3, 3>(0, 6) = Matrix3d::Identity() * dt;
        F.block<3, 3>(0, 9) = -0.25 * (delta_R + result_R) * dt * dt;
        F.block<3, 3>(0, 12) = 0.25 * result_R * R_a_1_x * dt * dt * dt;

        F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * dt;
        F.block<3, 3>(3, 12) = -Matrix3d::Identity() * dt;

        F.block<3, 3>(6, 3) = -0.5 * (delta_R * R_a_0_x + result_R * R_a_1_x * (Matrix3d::Identity() - R_w_x * dt)) * dt;
        F.block<3, 3>(6, 6) = Matrix3d::Identity();
        F.block<3, 3>(6, 9) = -0.5 * (delta_R + result_R) * dt;
        F.block<3, 3>(6, 12) = 0.5 * result_R * R_a_1_x * dt * dt;

        F.block<3, 3>(9, 9) = Matrix3d::Identity();
        F.block<3, 3>(12, 12) = Matrix3d::Identity();

        Eigen::Matrix<double, 15, 18> V;
        V.setZero();

        V.block<3, 3>(0, 0) = 0.25 * delta_R * dt * dt;
        V.block<3, 3>(0, 3) = -0.125 * result_R * R_a_1_x * dt * dt * dt;
        V.block<3, 3>(0, 6) = 0.25 * result_R * dt * dt;
        V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);

        V.block<3, 3>(3, 3) = 0.5 * Matrix3d::Identity() * dt;
        V.block<3, 3>(3, 9) = 0.5 * Matrix3d::Identity() * dt;

        V.block<3, 3>(6, 0) = 0.5 * delta_R * dt;
        V.block<3, 3>(6, 3) = -0.25 * result_R * R_a_1_x * dt * dt;
        V.block<3, 3>(6, 6) = 0.5 * result_R * dt;
        V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);

        V.block<3, 3>(9, 12) = Matrix3d::Identity() * dt;
        V.block<3, 3>(12, 15) = Matrix3d::Identity() * dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }

    inline Matrix3d skewSymmetric(const Vector3d &v) const
    {
        Matrix3d m;
        m << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
        return m;
    }

    Eigen::Vector3d acc_0, gyr_0;
    const Eigen::Vector3d linearized_acc, linearized_gyr;
    const double ACC_N_SQ, ACC_W_SQ, GYR_N_SQ, GYR_W_SQ;
    bool use_rk4;
    double dt;

    Eigen::Vector3d acc_1, gyr_1;

    Eigen::Matrix<double, 18, 18> noise;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;
};