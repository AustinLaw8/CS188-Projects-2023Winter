---
layout: post
comments: true
title: Driving Simulators
author: Victoria Lam, Austin Law
date: 2023-03-26
---

> Self-driving is a hot topic among deep vision learning. One way of training driving models is using imitation learning. In this work, we focus on reproducing the findings from “End-to-end Driving via Conditional Imitation Learning.” To do so, we utilize CARLA, a driving simulator, and emulate the models created in said paper using their provided dataset.

<!-- more -->
{: class="table-of-content"}
* Introduction
* Technical Details
* Issues Encountered
* References
{:toc} 

# Introduction
Our work explores the idea of conditional imitation learning model as experimented by “End-to-end Driving via Conditional Imitation Learning”, which addresses the limitations of imitation learning by proposing to condition imitation learning on high-level command input. This work will also draw ideas from “Conditional Affordance Learning for Driving in Urban Environments,” which also implements a similar method of using high-level directional inputs to improve autonomous driving. Imitation learning has not scaled up to fully autonomous urban driving due to limitations to its learning model. As explained in “End-to-end Driving via Conditional Imitation Learning”, one limitation is the assumption that the optimal action can be inferred from the perceptual input alone, however this does not hold in practice.

One method of dealing with this limitation is conditional imitation learning. We plan to first understand the standard imitation learning model. Then based on the original model, we will make modifications to conduct conditional imitation learning and compare the results from the original model.

In this post, we reiterate our initial project, the model design, and the model's results and shortcomings. We then describe the ways in which we attempted to fix our model, our failures, and what we believe to be the reasons for our failuers.

For a full project overview, see [this video](https://drive.google.com/file/d/1ZU-w0e3QYEljqy0RI9BLqVqQE2TLrVnB/view?usp=sharing).

# Technical Details

## CARLA Simulator
To explore and experiment with autonomous driving, we will be using CARLA (Car Learning to Act). CARLA is an open-source simulator implemented in Unreal Engine 4 used for autonomous driving research. CARLA has been developed from the ground up to support development, training, and validation of autonomous urban driving systems. CARLA provides open digital assets, such as urban layouts, buildings, vehicles, and pedestrians. The simulation platform supports flexible specification of sensor suites and environmental conditions.

![CARLA Simulator](/assets/images/team07/carla_2.jpg)
![Image Segmentation in CARLA](/assets/images/team07/carla_1.png)

## Imitation Learning Model
Imitation learning’s aim is to train the agent by demonstrating the desired behavior. Imitation learning is a form of supervised learning. The standard imitation learning model maps directly from raw input to control output and is learned from data in an end-to-end fashion. The data for imitation learning is collected through i.e. video footage and input monitoring of a human driver driving in the simulated environment.
The Data
The data is given as two separate but associated "datasets": a set of images and a set of vehicle measurements. While stored separately, they are associated based on the time step (captured at the same time). The image is a 200x88 image of the vehicle's front camera. Measurements include speed, acceleration, position, noise, etc.
The Model
The purpose of the model is to generate the set of actions for a given time step, described by the steering angle and acceleration (represented by magnitude of gas and brake). It takes as input the image and measurements of a time step and is expected to output three values: steering angle, gas amount, and brake amount.

The model has three main sections: a convolutional section and two fully connected sections. The image goes through the convolution section while the measurements are passed through one of the fully connected sections. Afterwards, the two results are concatenated and passed through a final fully connected section until the output is generated.

The convolutional section of the model utilizes convolutional layers to process images from the dataset. The image module consists of 8 convolutional and 2 fully connected layers. The convolution kernel size is 5 in the first layer and 3 in the following layers. The first, third, and fifth convolutional layers have a stride of 2. The first convolutional layer begins with 32 channels and increases to 256 channels in the last layer.

The fully-connected layers each contain 512 units. The measurements are passed through the fully connected sections. The speed of the car is used as the measurement. After all hidden layers, we used ReLU nonlinearities. 20% dropout is used after convolutional layers and 50% dropout is applied after fully-connected hidden layers.

For an image and code snippet of the architecture, see our previous post.

## Optimization
All models were trained using the Adam solver with learning rate `lr=.0002`.

The criterion used is MSELoss.

## Initial Results

After our first set of epochs of training over 3288 * 200 = 657600 rows of data, our model reached an average validation loss of .0911. However, ran into many issues testing in CARLA: namely, we had to implement our own driving agent to use the model, and implement various sensors so as to recieve the right measurements to pass to the model. After doing so, we quickly discovered that our model was not performing well. There are many possible reasons for this, but our two main conjectures are:
- We did not give the agent a real goal. With reinforcement learning, a driving agent would have a goal to drive from point a to point b without crashing, but in our imitation learning, the model is simply reacting to the environment based on the input. Because the streets are empty and the agent doesn't truly have to do anything, perhaps it learned that the best thing to do was nothing.
- It is difficult to know exactly what the data in the `measurements` input is supposed to reflect. There is little documentation in the paper/github we are basing our code off of with relation to what the data actually is; we are just extrapolating based on the name of the columns. Specifically, there were a few columns that we didn't understand: `Opposite Lane Inter` and `Sidewalk Intersect` being a few examples. 

[Old Model Demo](https://drive.google.com/file/d/1OAX7ifcuJxL4eo-GTiVYLUfDg8RY2g8o/view?usp=share_link)

## Further Results 

Given these issues and our failures, we decided to retrain our model, slightly differently. Originally, we watned to use all the columns in the dataset, since we believed that more data is better. However, the actual model in the code only uses a single column of the measurements: speed. Thus, we redesigned our model to use speed and speed only. This model reached a loss of .0300 after training. However, it ultimately performed about the same:

[New Model Demo](https://drive.google.com/file/d/1G_XMOH1Y4a2zU7HpAamFST-NpIDYuZYE/view?usp=share_link)

We believe that possible causes might be the same as above, in that the driving agent does not have a real goal. It may have learned to simply not move.

## Other Shortcomings

The original intention behind our project was to implement conditional imitation learning. Based on the `The Next Step` section of our last post, we intended to implement a final set of branches as the last layer of our model. Which branch is chosen would be selected by the control signal in the measurements. However, we ran into one main issue in attempting to do so: we did not know how to implement it. Mainly, we did not know how to calculate a loss function and implement the forward pass such that the backward pass worked. As a result, we were unable to implement this aspect of our project. We hoped that having this implemented would lead to better performance of our model, but whether this is true or not will be left unknown.

One final shortcoming was our understanding of the core aspect of conditional learning: the "command" that is passed in the measurements. However, implementing this command requires much more planning and finetuning, and we felt that, although important, may be out of scope due to time constraints. This goes back to the issue of our agent not having a goal: theoretically, the "goal" would be provided as a series of commands. For example, as described in the paper, a typical command would be to 'turn right at the next intersection', and at test time, the command would be provided by a 'human user or a planning module'.

# References

_End-to-end Driving via Conditional Imitation Learning._
Codevilla, Felipe and Müller, Matthias and López, Antonio and Koltun, Vladlen and Dosovitskiy, Alexey. ICRA 2018. https://arxiv.org/pdf/1710.02410.pdf

_CARLA: An Open Urban Driving Simulator_. Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, Vladlen Koltun; PMLR 78:1-16. https://arxiv.org/pdf/1806.06498.pdf
