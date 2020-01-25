/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: 48cfu
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * DONE: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * DONE: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if (!is_initialized) {
    num_particles = 10;  // DONE: Set the number of particles
    
    std::default_random_engine gen;
    // This line creates a normal (Gaussian) distribution for x,y,theta
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++){
      Particle p {};
      p.id = static_cast<int>(i) + 1;
      p.x = dist_x(gen);
      p.y = dist_y(gen); 
      p.theta = dist_theta(gen);
      p.weight = 1.0 / num_particles;
      particles.push_back(p); //migliorare con emplace_back
      weights.push_back(p.weight);
    }
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * DONE: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  if (!ParticleFilter::initialized()) throw std::runtime_error("Prediction called without initialization.");
  std::default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x,y,theta
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  for (size_t i = 0; i < particles.size(); i++){
    double x0 = particles[i].x;
    double y0 = particles[i].y;
    double theta0 = particles[i].theta;
    particles[i].x = x0 + (velocity/yaw_rate) * (sin(theta0 + yaw_rate * delta_t) - sin(theta0)) +  dist_x(gen);
    particles[i].y = y0 + (velocity/yaw_rate) * (cos(theta0) - cos(theta0 + yaw_rate * delta_t)) +  dist_y(gen);
    particles[i].theta = theta0 + yaw_rate * delta_t + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * CHECK: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  if (!ParticleFilter::initialized()) throw std::runtime_error("dataAssociation called without initialization.");
  
  /* First need to transform the car's measurements from its local 
  * car coordinate system to the map's coordinate system
  * "observations" are in car coordinate system
  */

  /* Next, each measurement will need to be associated with a landmark identifier, 
  * for this part we will take the closest landmark to each transformed observation
  */

  //O(nm)
  //std::cout << "Size" << predicted.size() << std::endl << std::flush;
  for (size_t i = 0; i < observations.size() && predicted.size() > 0; i++){
    double distance = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
    size_t minimum_index = 0;
    for (size_t j = 0; j < predicted.size(); j++){
      double new_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (new_distance <= distance){
        distance = new_distance;
        minimum_index = j;
      }
    }
    // associate observations to closest landmarks in predicted vector
    observations[i].id = predicted[minimum_index].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  //std::cout << "In updateWeights" << std::endl << std::flush;
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  if (!ParticleFilter::initialized()) throw std::runtime_error("updateWeights called without initialization.");

  std::default_random_engine gen;
  // This line creates a normal (Gaussian) distribution for x,y,theta
  std::normal_distribution<double> dist_x(0, std_landmark[0]);
  std::normal_distribution<double> dist_y(0, std_landmark[1]);

  
  // Observations from car coordinate to map cordinate system (for each particle)
  std::vector<LandmarkObs> observations_map_coordinates (observations.size(), LandmarkObs());

  for (size_t i = 0; i < particles.size(); i++){
    double theta = particles[i].theta;
    // transform observations into map coordinate
    for (size_t j = 0; j < observations.size(); j++){
      observations_map_coordinates[j].id = observations[j].id;
      observations_map_coordinates[j].x = particles[i].x + (cos(theta) * observations[j].x) - (sin(theta) * observations[j].y);
      observations_map_coordinates[j].y = particles[i].y + (sin(theta) * observations[j].x) + (cos(theta) * observations[j].y);
    }
    //std::cout << "In 1" << std::endl << std::flush;
    // predict landmarks withing sensor range
    std::vector<LandmarkObs> predicted{};
    for (size_t j = 0; j < map_landmarks.landmark_list.size(); j++){
      LandmarkObs temp;
      temp.id = map_landmarks.landmark_list[j].id_i;
      temp.x = map_landmarks.landmark_list[j].x_f + dist_x(gen);
      temp.y = map_landmarks.landmark_list[j].y_f + dist_y(gen);
      // take only landmarks within sensor range
      //if (dist(particles[i].x, particles[i].y, temp.x, temp.y) <= sensor_range)
        predicted.push_back(temp);
    }
    // associate observations to landmarks using nearest landmark
    dataAssociation(predicted, observations_map_coordinates);
    // Update particle weight

    double new_weight = 1;
    for (size_t j = 0; j < observations_map_coordinates.size(); j++){
      int closest_landmark = observations_map_coordinates[j].id - 1; // landmarks ID start from 1
      double mu_x = map_landmarks.landmark_list[closest_landmark].x_f;
      double mu_y = map_landmarks.landmark_list[closest_landmark].y_f;
      new_weight *= multiv_prob(std_landmark[0], std_landmark[1], observations_map_coordinates[j].x, observations_map_coordinates[j].y, mu_x, mu_y);
    }
    weights[i] = new_weight;
  }

  // normalize
  weights = normalize_vector(weights);
  for (size_t i = 0; i < weights.size(); i++){
    particles[i].weight = weights[i];
    std::cout << weights[i] << " ";
  }
  std::cout << std::endl << std::flush;
}

void ParticleFilter::resample() {
  /**
   * DONE: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  if (!ParticleFilter::initialized()) throw std::runtime_error("resample called without initialization.");
  std::default_random_engine gen;
  std::discrete_distribution<int> samples(weights.begin(), weights.end());
  std::vector<double> new_weights (weights.size(), 0);
  for (size_t i = 0; i < weights.size(); i++){
    size_t index = samples(gen);
    new_weights[index]++;
  }
  
  for (size_t i = 0; i < weights.size(); i++){
    weights[i] = new_weights[i] / static_cast<double>(weights.size());
    particles[i].weight = weights[i];
  }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}