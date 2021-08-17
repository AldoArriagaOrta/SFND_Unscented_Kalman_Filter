#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  //set state vector's dimension
  n_x_ = 5;

  //set augmented state vector's dimension
  n_aug_ = 7;

  // set lidar measurement vector's dimension
  n_zl_ = 2;

  // set radar measurement vector's dimension
  n_zr_ = 3;

  // state vector
  x_ = VectorXd(n_x_);

  // augmented mean state vector
  x_aug_ = VectorXd(n_aug_);

  // state covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // augmented state covariance matrix
  P_aug_ = MatrixXd(n_aug_, n_aug_);

  // augmented sigma points matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;//30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.45;//30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // vector for sigma points weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  
  // process noise matrix
  Q_ = MatrixXd(2, 2);
  Q_ << std_a_ * std_a_, 0, 
		0, std_yawdd_ *std_yawdd_;

  // Lidar measurement noise matrix
  R_lidar_ = MatrixXd(n_zl_, n_zl_);
  R_lidar_ <<	std_laspx_ * std_laspx_, 0,
				0, std_laspy_ * std_laspy_;

  //Radar measurment noise matrix
  R_radar_ = MatrixXd(n_zr_, n_zr_);
  R_radar_ <<	std_radr_ * std_radr_, 0, 0,
				0, std_radphi_ *std_radphi_, 0,
				0, 0, std_radrd_ * std_radrd_;
 
  nlsq_aug_ = sqrt(n_aug_ + lambda_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	if (!is_initialized_) {

		// Initialization of state vector x_ with first measurement
		x_ << 0, 0, 0, 0, 0; // px, py, v, yaw, yaw_dot

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

			// Initialization for radar measurement
			// mapping polar radar measurements (rho, phi) to state variables px, py
			x_ <<	meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]),
					meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]),
					0,
					0,
					0;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			// Initialization for lidar measurement
			// set the state with the initial location and zero velocity
			x_ <<	meas_package.raw_measurements_[0], 
					meas_package.raw_measurements_[1], 
					0, 
					0,
					0;
		}

		// initial state covariance matrix P
		P_ <<	0.15, 0, 0, 0, 0,
				0, 0.15, 0, 0, 0,
				0, 0, 0.5, 0, 0,
				0, 0, 0, 0.5, 0,
				0, 0, 0, 0, 2;

		// first timestamp is needed for next cycle
		previous_timestamp_ = meas_package.timestamp_;
		
		is_initialized_ = true;
		return;
	}

	// compute the time elapsed between the current and previous measurements
	double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//delta t in seconds
	previous_timestamp_ = meas_package.timestamp_;
	
	Prediction(dt);
	Update(meas_package);

	// print the output
	//cout << "x_ = " << x_ << endl;
	//cout << "P_ = " << P_ << endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

	/****************************************************
	* 1. State Augmentation and Sigma Points Generation	*
	*****************************************************/

	x_aug_.fill(0.0);
	x_aug_.head(n_x_) = x_;

	// populate augmented covariance matrix
	P_aug_.fill(0.0);
	P_aug_.topLeftCorner(n_x_, n_x_) = P_;
	P_aug_.bottomRightCorner(2, 2) = Q_;

	// create square root of P matrix
	MatrixXd A_aug_ = P_aug_.llt().matrixL();

	// compute sigma points
	// set sigma points as columns of matrix Xsig_aug
	Xsig_aug_.fill(0.0);
	Xsig_aug_.col(0) = x_aug_;

	for (int i = 0; i<n_aug_; i++){
		Xsig_aug_.col(i + 1) = x_aug_ + nlsq_aug_ * A_aug_.col(i);
		Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - nlsq_aug_ * A_aug_.col(i);
	}

	/****************************************************
	*			2. Sigma Points Prediction				*
	*****************************************************/

	//predict sigma points
	double dt_sq = delta_t * delta_t;

	Xsig_pred_.fill(0.0);

	for (int i = 0; i<2 * n_aug_ + 1; i++){

		//extract sigma points
		double x = Xsig_aug_(0, i);
		double y = Xsig_aug_(1, i);
		double v = Xsig_aug_(2, i);
		double yaw = Xsig_aug_(3, i);
		double yawd = Xsig_aug_(4, i);
		double nu_a = Xsig_aug_(5, i);
		double nu_yaw_ac = Xsig_aug_(6, i);
		double x_p, y_p, v_p, yaw_p, yawd_p;

		//CTRV model
		v_p = v + delta_t * nu_a;
		yaw_p = yaw + yawd * delta_t + 0.5 * dt_sq *nu_yaw_ac;
		yawd_p = yawd + delta_t * nu_yaw_ac;

		double cos_yaw = cos(yaw);
		double sin_yaw = sin(yaw);

		if (fabs(yawd)>0.0001){
			x_p = x + (v / yawd) * (sin(yaw + yawd * delta_t) - sin_yaw) + 0.5 * dt_sq * cos_yaw * nu_a;
			y_p = y + (v / yawd) * (cos_yaw - cos(yaw + yawd * delta_t)) + 0.5 * dt_sq * sin_yaw * nu_a;
		}
		else{
			x_p = x + v * cos_yaw * delta_t + 0.5 * dt_sq * cos_yaw * nu_a;
			y_p = y + v * sin_yaw * delta_t + 0.5 * dt_sq * sin_yaw * nu_a;
		}

		// put in corresponding matrix columns
		Xsig_pred_(0, i) = x_p;
		Xsig_pred_(1, i) = y_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}

	/****************************************************
	*		3. Predicted State Mean and Covariance		*
	*****************************************************/

	//predict state mean
	x_.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		x_ = x_+ weights_(i) * Xsig_pred_.col(i);
	}

	//predict state covariance matrix
	P_.fill(0.0);

	for (int i = 0; i<2 * n_aug_ + 1; i++)
	{
		VectorXd Xdiff = Xsig_pred_.col(i) - x_;
		Xdiff(3) = atan2(sin(Xdiff(3)), cos(Xdiff(3)));// angle normalization
		P_ = P_ + weights_(i) * Xdiff * Xdiff.transpose();
	}
}

void UKF::Update(MeasurementPackage meas_package) {

	//check for the type of sensor
	bool is_radar;
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		// radar updates
		is_radar = true;
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		// lidar updates
		is_radar = false;
	}
	else {
		return;
	}

	MatrixXd Zsig ; //matrix for measurement sigma points 
	VectorXd z_pred; //predicted  measurements
	MatrixXd S ; // innovation covariance matrix 
	VectorXd z ; // vector  measurements
	MatrixXd Tc ; // cross correlation matrix 

	// set matrices' dimensions depending on type of measurement
	if (is_radar) {
		 Zsig = MatrixXd(n_zr_, 2 * n_aug_ + 1); 
		 z_pred = VectorXd(n_zr_); 
		 S = MatrixXd(n_zr_, n_zr_);
		 z = VectorXd(n_zr_); 
		 Tc = MatrixXd(n_x_, n_zr_); 
	}
	else {
		 Zsig = MatrixXd(n_zl_, 2 * n_aug_ + 1); 
		 z_pred = VectorXd(n_zl_); 
		 S = MatrixXd(n_zl_, n_zl_);
		 z = VectorXd(n_zl_);
		 Tc = MatrixXd(n_x_, n_zl_); 
	}

	Zsig.setZero();
	z_pred.setZero();
	S.setZero();
	z.setZero();
	Tc.setZero();

	/****************************************************
	*				1. Predict Measurement				*
	*****************************************************/

	// transform sigma points computed in Prediction phase into the appropriate measurement space (radar or lidar)
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		double px = Xsig_pred_(0, i);
		double py = Xsig_pred_(1, i);

		if (is_radar) {
			double v = Xsig_pred_(2, i);
			double yaw = Xsig_pred_(3, i);
			double mag = sqrt(px*px + py*py);
			Zsig(0, i) = mag;
			Zsig(1, i) = atan2(py, px);
			// check division by zero
			if (mag > 0) {
				Zsig(2, i) = ((px* cos(yaw) *v) + (py* sin(yaw) * v)) / mag;
			}
			else {
				Zsig(2, i) = 0;
			}
		}
		else {
			Zsig(0, i) = px;
			Zsig(1, i) = py;
		}
	}

	// compute mean predicted measurement
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	// compute innovation covariance matrix S	
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd Zdiff = Zsig.col(i) - z_pred;
		if (is_radar) {
			Zdiff(1) = atan2(sin(Zdiff(1)), cos(Zdiff(1)));// angle normalization
		}
		S = S + weights_(i) * Zdiff * Zdiff.transpose();
	}

	if (is_radar) {
		S = S + R_radar_;
	}
	else
	{
		S = S + R_lidar_;
	}

	/****************************************************
	*					2. Update						*
	*****************************************************/

	// extract measurement package
	if (is_radar) {
		z <<
			meas_package.raw_measurements_[0],  //rho in m
			meas_package.raw_measurements_[1],  //phi in rad
			meas_package.raw_measurements_[2];	//rho_dot in m/s
	}
	else {
		z <<
			meas_package.raw_measurements_[0],   //measured px
			meas_package.raw_measurements_[1];	 //measured py
	}

	// compute cross correlation matrix Tc
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd state_diff = Xsig_pred_.col(i) - x_;
		VectorXd meas_diff = Zsig.col(i) - z_pred;
		state_diff(3) = atan2(sin(state_diff(3)), cos(state_diff(3)));
		if (is_radar) {
			meas_diff(1) = atan2(sin(meas_diff(1)), cos(meas_diff(1)));
		}
		Tc = Tc + weights_(i) * state_diff * meas_diff.transpose();
	}

	// compute Kalman gain K;
	MatrixXd K = Tc * S.inverse();
	VectorXd y = z - z_pred;
	if (is_radar) {
		y(1) = atan2(sin(y(1)), cos(y(1)));
	}

	// update state and covariance matrix
	x_ = x_ + K *y;
	P_ = P_ - K * S * K.transpose();

	if (is_radar) {
		NIS_radar_ = y.transpose() * S.inverse() * y;
		cout << "NIS_radar=" << NIS_radar_ << endl;

		ofstream myfile;
		myfile.open("NIS_radar.csv", std::ios_base::app | std::ios_base::out);
		if (myfile.is_open())
		{
			myfile << (meas_package.timestamp_) / 1000000.0<<", "<< NIS_radar_<< "\n";
			myfile.close();
		}
		else cout << "Unable to open file";
		return;
	}
	else {
		NIS_lidar_ = y.transpose() * S.inverse() * y;
		cout << "NIS_lidar=" << NIS_lidar_ << endl;

		ofstream myfile;
		myfile.open("NIS_lidar.csv", std::ios_base::app | std::ios_base::out);
		if (myfile.is_open())
		{
			myfile << (meas_package.timestamp_) / 1000000.0 << ", " << NIS_lidar_ << "\n";
			myfile.close();
		}
		else cout << "Unable to open file";
		return;
	}
}

// void UKF::UpdateLidar(MeasurementPackage meas_package) {
//   /**
//    * TODO: Complete this function! Use lidar data to update the belief 
//    * about the object's position. Modify the state vector, x_, and 
//    * covariance, P_.
//    * You can also calculate the lidar NIS, if desired.
//    */
// }

// void UKF::UpdateRadar(MeasurementPackage meas_package) {
//   /**
//    * TODO: Complete this function! Use radar data to update the belief 
//    * about the object's position. Modify the state vector, x_, and 
//    * covariance, P_.
//    * You can also calculate the radar NIS, if desired.
//    */
// }