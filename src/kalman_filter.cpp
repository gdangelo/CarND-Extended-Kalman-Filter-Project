#include "kalman_filter.h"
#include "tools.h"

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
  VectorXd Ft = F_.transpose();
  P_ = F_*P_*Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_*x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_*Ht*Si;
  MatrixXd I = MatrixXd::Identity(4, 4);

  // Update
  x_ = x_ + K*y;
  P_ = (I - K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Use the equations that map the predicted location
  // from Cartesian coordinates to polar coordinates
  VectorXd hx(3);
  hx << sqrt(x_(0)*x_(0) + x_(1)*x_(1)),
                 atan(x_(1)/x_(0)),
                 (x_(0)*x_(2) + x_(1)*x_(3))/hx(0);

  VectorXd y = z - hx;

  Tools tools;
  MatrixXd Hj = tools.CalculateJacobian(x_);
  MatrixXd Ht = Hj.transpose();
  MatrixXd S = Hj*P_*Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_*Ht*Si;
  MatrixXd I = MatrixXd::Identity(4, 4);

  // Update
  x_ = x_ + K*y;
  P_ = (I - K*Hj)*P_;
}
