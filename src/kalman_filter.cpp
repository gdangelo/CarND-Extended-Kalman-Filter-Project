#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Predict the state after delta_T
  x_ = F_*x_;
  // Update the State Covariance Matrix P (uncertainty)
  MatrixXd Ft = F_.transpose();
  P_ = F_*P_*Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_*x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_*Ht*Si;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  // Update
  x_ = x_ + K*y;
  P_ = (I - K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Use the equations that map the predicted location
  // from Cartesian coordinates to polar coordinates
  VectorXd hx(3);

  float ro = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  float theta = atan2(x_(1), x_(0));
  float ro_dot;
  if (fabs(ro) < 0.0001){
    ro_dot = 0;
  }
  else{
    ro_dot = (x_(0)*x_(2) + x_(1)*x_(3))/ro;
  }
  hx << ro, theta, ro_dot;

  VectorXd y = z - hx;
  // Angle normalization --> [-pi; pi]
  y[1] = atan2(sin(y[1]), cos(y[1]));

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_*Ht;
  MatrixXd K = PHt*Si;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  // Update
  x_ = x_ + K*y;
  P_ = (I - K*H_)*P_;
}
