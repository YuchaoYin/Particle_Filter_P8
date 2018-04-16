/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // number of particles
    num_particles = 50;
    // standard deviations for x, y and theta : std[0], std[1], std[2]
    // create normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // generate particles
    for (int i = 0; i < num_particles; ++i){
        Particle P;
        P.id = i;
        P.x = dist_x(gen);
        P.y = dist_y(gen);
        P.theta = dist_theta(gen);
        P.weight = 1.0;

        particles.push_back(P);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];
    normal_distribution<double> dist_x(0.0, std_x*delta_t);
    normal_distribution<double> dist_y(0.0, std_y*delta_t);
    normal_distribution<double> dist_theta(0.0, std_theta*delta_t);

    for (Particle &particle : particles) {
            if (fabs(yaw_rate) == 0){
                particle.x += velocity*delta_t*cos(particle.theta) + dist_x(gen);
                particle.y += velocity*delta_t*sin(particle.theta) + dist_y(gen);
                particle.theta= particle.theta + dist_theta(gen);
            }
            else{
                particle.x += (velocity/yaw_rate)*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta)) + dist_x(gen);
                particle.y += (velocity/yaw_rate)*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t)) + dist_y(gen);
                particle.theta += yaw_rate*delta_t + dist_theta(gen);
            }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    double min = numeric_limits<double>::max();
    for (auto& observation : observations){
        for (const auto& prd : predicted){
            double distance  = dist(observation.x, observation.y, prd.x, prd.y);
            if (distance < min){
                observation.id = prd.id;
                min = distance;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // predict measurements to all the map landmarks within sensor range for each particle
    for (int i = 0; i < num_particles; ++i){

        double x_p = particles[i].x;
        double y_p = particles[i].y;
        double theta_p = particles[i].theta;

        std::vector<LandmarkObs> predicted;

        for (const auto& map_landmark : map_landmarks.landmark_list){
            double x_l = map_landmark.x_f;
            double y_l = map_landmark.y_f;
            int id_l = map_landmark.id_i;

            double d = dist(x_p, y_p, x_l, y_l);

            if (d < sensor_range){
                LandmarkObs l;
                l.x = x_l;
                l.y = y_l;
                l.id = id_l;
               // cout << "id" << l.id << endl;
                predicted.push_back(l);
            }
        }

        // map car coordinates to map coordinates
        std::vector<LandmarkObs> transformed_obss;
        for (int j = 0; j < observations.size(); ++j){
            LandmarkObs transformed_obs;
            transformed_obs.x = cos(theta_p)*observations[j].x - sin(theta_p)*observations[j].y + x_p;
            transformed_obs.y = sin(theta_p)*observations[j].x + cos(theta_p)*observations[j].y + y_p;
            transformed_obss.push_back(transformed_obs);
        }

        // associating these transformed observations with the neariest landmark on the map
        dataAssociation(predicted, transformed_obss);

        // calculate particle weight
        double particle_weight = 1.0;
        double mu_x, mu_y;
        double stddev_x = std_landmark[0];
        double stddev_y = std_landmark[1];

        for (const auto& obs : transformed_obss){
            for (const auto& pred : predicted){
                if (pred.id == obs.id){
                    mu_x = pred.x;
                    mu_y = pred.y;
                    break;
                }
            }
            //cout << "mu_y" << mu_x << endl;
            //cout << "obs.y" << obs.x << endl;
            double norm_factor = 2.0 * M_PI * stddev_x * stddev_y;
            //cout << "norm_factor" << norm_factor << endl;
            double prob = exp( -( pow(obs.x-mu_x,2)/pow(stddev_x,2) + pow(obs.y-mu_y,2)/pow(stddev_y,2)));
            //cout << "prob" << prob <<endl;
            long double multipler = prob/norm_factor;
            if (multipler > 0){
                particle_weight *= multipler;
            }
        }
        //cout << "weight: " << particle_weight<< endl;
        particles[i].weight = particle_weight;
    }

    // normalization
    double sum = 0.0;
    for (const auto& particle : particles){
        sum += particle.weight;
    }

    cout<< "sum" << sum << endl;
    for (auto& particle : particles){
        particle.weight /= (sum + numeric_limits<double>::epsilon());
    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<double> weights;
    for (const auto& particle : particles){

        //cout<< "weight" << particle.weight<< endl;
        weights.push_back(particle.weight);
    }
    std::vector<Particle> resampled_particles;
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    for (int i = 0; i < num_particles; ++i){
        resampled_particles.push_back(particles[d(gen)]);
    }
    particles = resampled_particles;

    for (auto& particle : particles){
        particle.weight = 1.0;
    }
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
