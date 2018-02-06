#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Initialize rmse vector
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
  if(estimations.size() == 0 || estimations.size() != ground_truth.size()){
    cout << "CalculateRMSE() - Error - Invalid inputs." << endl;
    return rmse;
  }

  // Accumulate squared residuals
  for(int i = 0; i < estimations.size(); i++){
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array(); //coefficient-wise multiplication
    rmse += residual;
  }

  // Compute the mean
  rmse = rmse/estimations.size();

  // Compute the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  // Initialize Hj matrix
  MatrixXd Hj(3, 4);

  // Get state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float d1 = px*px + py*py;
  float d2 = sqrt(d1);
  float d3 = d1*d2;

  // Make sure we don't divide by zero
  if(d1 == 0){
    cout << "CalculateJacobian() - Error - Divison by zero." << endl;
  }
  // Compute Jacobian matrix
  else{
    Hj << px/d2, py/d2, 0, 0,
          -py/d1, px/d1, 0, 0,
          py*(vx*py - vy*px), px*(vy*px - vx*py), px/d2, py/d2;
  }

  return Hj;
}
