/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "loop_fusion/keyframe.hpp"
#include "loop_fusion/utility/Grider_FAST.hpp"

#include <cstring>
#include <utility>

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
	int j = 0;
	for (int i = 0; i < int(v.size()); i++)
		if (status[i])
			v[j++] = v[i];
	v.resize(j);
}

template <typename... Args>
static void reduceVectors(const std::vector<unsigned char> &status, Args &...vectors)
{
	const size_t N = status.size();
	if (((vectors.size() != N) || ...))
		return;

	size_t j = 0;
	for (size_t i = 0; i < N; ++i)
	{
		if (status[i])
		{
			if (j != i)
			{
				((vectors[j] = std::move(vectors[i])), ...);
			}
			++j;
		}
	}

	((vectors.resize(j)), ...);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
				   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
				   vector<double> &_point_id, int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	if (!USE_DL_LOOP)
	{
		computeWindowBRIEFPoint();
		computeBRIEFPoint();
	}
	if (!DEBUG_IMAGE)
		image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
				   cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1> &_loop_info,
				   vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	// vio_T_w_i = _vio_T_w_i;
	// vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image = _image.clone();
		cv::resize(image, thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}

void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for (int i = 0; i < (int)point_2d_uv.size(); i++)
	{
		cv::KeyPoint key;
		key.pt = point_2d_uv[i];
		window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	const int fast_th = 10; // corner detector response threshold
	Grider_FAST::perform_griding(image, keypoints, 200, 1, 1, fast_th, true);
	for (int i = 0; i < (int)point_2d_uv.size(); i++)
	{
		cv::KeyPoint key;
		key.pt = point_2d_uv[i];
		keypoints.push_back(key);
	}

	extractor(image, keypoints, brief_descriptors);
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
		keypoints_norm.push_back(tmp_norm);
	}
}

void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
	m_brief.compute(im, keys, descriptors);
}

bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
							const std::vector<BRIEF::bitset> &descriptors_old,
							const std::vector<cv::KeyPoint> &keypoints_old,
							const std::vector<cv::KeyPoint> &keypoints_old_norm,
							cv::Point2f &best_match,
							cv::Point2f &best_match_norm)
{
	cv::Point2f best_pt;
	int bestDist = 128;
	int bestIndex = -1;
	for (int i = 0; i < (int)descriptors_old.size(); i++)
	{

		int dis = HammingDis(window_descriptor, descriptors_old[i]);
		if (dis < bestDist)
		{
			bestDist = dis;
			bestIndex = i;
		}
	}
	if (bestIndex != -1 && bestDist < 80)
	{
		best_match = keypoints_old[bestIndex].pt;
		best_match_norm = keypoints_old_norm[bestIndex].pt;
		return true;
	}
	else
		return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
								std::vector<uchar> &status,
								const std::vector<BRIEF::bitset> &descriptors_old,
								const std::vector<cv::KeyPoint> &keypoints_old,
								const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
	for (int i = 0; i < (int)window_brief_descriptors.size(); i++)
	{
		cv::Point2f pt(0.f, 0.f);
		cv::Point2f pt_norm(0.f, 0.f);
		if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
			status.push_back(1);
		else
			status.push_back(0);
		matched_2d_old.push_back(pt);
		matched_2d_old_norm.push_back(pt_norm);
	}
}

void KeyFrame::attachDLFeatures(const std::unordered_map<int, std::vector<float>> &id_to_desc)
{
	const int n = (int)point_id.size();
	dl_valid.assign(n, 0);
	if (n == 0 || id_to_desc.empty())
		return;

	// Dimensionality comes from the incoming descriptors
	const int dim = (int)id_to_desc.begin()->second.size();
	if (dim <= 0)
		return;

	dl_descriptors = cv::Mat::zeros(n, dim, CV_32F);
	for (int i = 0; i < n; i++)
	{
		auto it = id_to_desc.find((int)point_id[i]);
		if (it == id_to_desc.end() || (int)it->second.size() != dim)
			continue;
		std::memcpy(dl_descriptors.ptr<float>(i), it->second.data(),
					dim * sizeof(float));
		dl_valid[i] = 1;
	}
}

void KeyFrame::searchByDLDes(std::vector<cv::Point2f> &matched_2d_old,
							 std::vector<cv::Point2f> &matched_2d_old_norm,
							 std::vector<uchar> &status,
							 KeyFrame *old_kf)
{
	const int n_cur = (int)point_2d_uv.size();
	const int n_old = (int)old_kf->point_2d_uv.size();
	const int dim = dl_descriptors.empty() ? 0 : dl_descriptors.cols;
	const bool old_ok = !old_kf->dl_descriptors.empty() &&
						old_kf->dl_descriptors.cols == dim;

	for (int i = 0; i < n_cur; i++)
	{
		cv::Point2f pt(0.f, 0.f), pt_norm(0.f, 0.f);
		bool ok = false;
		if (dim > 0 && old_ok && i < (int)dl_valid.size() && dl_valid[i])
		{
			const float *cur = dl_descriptors.ptr<float>(i);
			float best = -2.f, second = -2.f;
			int best_idx = -1;
			for (int j = 0; j < n_old; j++)
			{
				if (j >= (int)old_kf->dl_valid.size() || !old_kf->dl_valid[j])
					continue;
				const float *o = old_kf->dl_descriptors.ptr<float>(j);
				float dot = 0.f;
				for (int k = 0; k < dim; k++)
					dot += cur[k] * o[k];
				if (dot > best)
				{
					second = best;
					best = dot;
					best_idx = j;
				}
				else if (dot > second)
					second = dot;
			}
			// Lowe ratio test on cosine distance (1 - similarity), plus a mild
			// absolute-similarity floor to reject obvious non-matches.
			if (best_idx >= 0 && best > 0.2f &&
				(1.f - best) < DL_RATIO_TEST * (1.f - second))
			{
				ok = true;
				pt = old_kf->point_2d_uv[best_idx];
				pt_norm = old_kf->point_2d_norm[best_idx];
			}
		}
		status.push_back(ok ? 1 : 0);
		matched_2d_old.push_back(pt);
		matched_2d_old_norm.push_back(pt_norm);
	}
}

void KeyFrame::FundamentalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
									  const std::vector<cv::Point2f> &matched_2d_old_norm,
									  vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
	if (n >= 8)
	{
		vector<cv::Point2f> tmp_cur(n), tmp_old(n);
		for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
		{
			double tmp_x, tmp_y;
			tmp_x = FOCAL_LENGTH_PX * matched_2d_cur_norm[i].x + COL / 2.0;
			tmp_y = FOCAL_LENGTH_PX * matched_2d_cur_norm[i].y + ROW / 2.0;
			tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

			tmp_x = FOCAL_LENGTH_PX * matched_2d_old_norm[i].x + COL / 2.0;
			tmp_y = FOCAL_LENGTH_PX * matched_2d_old_norm[i].y + ROW / 2.0;
			tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
		}
		cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
	}
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
						 const std::vector<cv::Point3f> &matched_3d,
						 std::vector<uchar> &status,
						 Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
	// for (int i = 0; i < matched_3d.size(); i++)
	//	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
	cv::Mat r, rvec, t, D, tmp_r;
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
	Matrix3d R_inital;
	Vector3d P_inital;
	Matrix3d R_w_c = origin_vio_R * qic;
	Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

	R_inital = R_w_c.inverse();
	P_inital = -(R_inital * T_w_c);

	cv::eigen2cv(R_inital, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_inital, t);

	cv::Mat inliers;

	int flags = cv::SOLVEPNP_EPNP;
	if (CV_MAJOR_VERSION < 3)
		solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 200, PNP_INFLATION / max_focallength, 100, inliers, flags);
	else
	{
		if (CV_MINOR_VERSION < 2)
			solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 200, sqrt(PNP_INFLATION / max_focallength), 0.99, inliers, flags);
		else
			solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 200, PNP_INFLATION / max_focallength, 0.99, inliers, flags);
	}

	for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
		status.push_back(0);

	for (int i = 0; i < inliers.rows; i++)
	{
		int n = inliers.at<int>(i);
		status[n] = 1;
	}

	cv::Rodrigues(rvec, r);
	Matrix3d R_pnp, R_w_c_old;
	cv::cv2eigen(r, R_pnp);
	R_w_c_old = R_pnp.transpose();
	Vector3d T_pnp, T_w_c_old;
	cv::cv2eigen(t, T_pnp);
	T_w_c_old = R_w_c_old * (-T_pnp);

	PnP_R_old = R_w_c_old * qic.transpose();
	PnP_T_old = T_w_c_old - PnP_R_old * tic;
}

bool KeyFrame::findConnection(KeyFrame *old_kf)
{
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	if (USE_DL_LOOP)
		searchByDLDes(matched_2d_old, matched_2d_old_norm, status, old_kf);
	else
		searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status,
						 old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
	reduceVectors(status, matched_2d_cur, matched_2d_old, matched_2d_cur_norm,
				  matched_2d_old_norm, matched_3d, matched_id);

	status.clear();
	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		status.clear();
		PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
		reduceVectors(status, matched_2d_cur, matched_2d_old, matched_2d_cur_norm,
					  matched_2d_old_norm, matched_3d, matched_id);
		if (DEBUG_IMAGE)
		{
			int gap = 10;
			cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
			cv::Mat gray_img, loop_match_img;
			cv::Mat old_img = old_kf->image;
			cv::hconcat(image, gap_image, gap_image);
			cv::hconcat(gap_image, old_img, gray_img);
			cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
			for (int i = 0; i < (int)matched_2d_cur.size(); i++)
			{
				cv::Point2f cur_pt = matched_2d_cur[i];
				cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
			}
			for (int i = 0; i < (int)matched_2d_old.size(); i++)
			{
				cv::Point2f old_pt = matched_2d_old[i];
				old_pt.x += (COL + gap);
				cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
			}
			for (int i = 0; i < (int)matched_2d_cur.size(); i++)
			{
				cv::Point2f old_pt = matched_2d_old[i];
				old_pt.x += (COL + gap);
				cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
			}
			cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
			putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

			putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
			cv::vconcat(notation, loop_match_img, loop_match_img);

			if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
			{
				cv::Mat thumbimage;
				cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
				// sensor_msgs::msg::ImagePtr
				sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", thumbimage).toImageMsg();

				int sec_ts = (int)time_stamp;
				uint nsec_ts = (uint)((time_stamp - sec_ts) * 1e9);
				msg->header.stamp.sec = sec_ts;
				msg->header.stamp.nanosec = nsec_ts;

				pub_match_img->publish(*msg);
			}
		}
	}

	// Record final inlier count for metrics regardless of acceptance outcome.
	last_inlier_count = static_cast<int>(matched_2d_cur.size());

	if (last_inlier_count > MIN_LOOP_NUM)
	{
		relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
		relative_q = PnP_R_old.transpose() * origin_vio_R;
		relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
		if (abs(relative_yaw) < MAX_THETA_DIFF && relative_t.norm() < MAX_POS_DIFF)
		{

			has_loop = true;
			loop_index = old_kf->index;
			loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
				relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
				relative_yaw;
			return true;
		}
	}
	return false;
}

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
	BRIEF::bitset xor_of_bitset = a ^ b;
	int dis = xor_of_bitset.count();
	return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
	_T_w_i = vio_T_w_i;
	_R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
	_T_w_i = T_w_i;
	_R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
	return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
	return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
	return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1> &_loop_info)
{
	if (abs(_loop_info(7)) < MAX_THETA_DIFF && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < MAX_POS_DIFF)
	{
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
	// loads the pattern
	cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
	if (!fs.isOpened())
		throw string("Could not open file ") + pattern_file;

	vector<int> x1, y1, x2, y2;
	fs["x1"] >> x1;
	fs["x2"] >> x2;
	fs["y1"] >> y1;
	fs["y2"] >> y2;

	m_brief.importPairs(x1, y1, x2, y2);
}
