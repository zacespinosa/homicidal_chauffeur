# Homicidal Chauffeur
By Zac Espinosa and Tarun Punnoose

This repo is interested in solving the homocidal chauffer problem using reinforcement learning. 

The [homicidal chauffeur problem](https://en.wikipedia.org/wiki/Homicidal_chauffeur_problem) is a canonical two person differential game from the 1950’s where a runner, that is highly maneuverable but slow, must survive the attacks of a homicidal chauffeur driver that is faster, but less maneuverable. The runner’s objective is to maximize the amount of time that it can survive in a continuous, bounded space, while the driver’s objective is to hit the runner in the shortest amount of time. The continuous state space in which the runner and driver play may contain movable and/or immovable obstacles, and, the runner can be modeled with or without fatigue. The car is modeled as an inertial Dubin’s car model with some minimum turning radius, acceleration and braking speed, and the runner is modeled as inertialess with the ability to instantaneously change velocity.
