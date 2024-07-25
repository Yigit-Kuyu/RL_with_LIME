## INFO

First, run  `SAC_Main.py` , key points are here:
- **State Representation:** The state is represented by features such as x, y, vx, and vy for each vehicle.

- **Action Representation:** The actions are represented by action_speed and action_steering_angle.

- **Interaction with Environment:** The agent interacts with the environment using the actor model to decide actions and collects the resulting state and action data.

- **Data Storage:** The collected data is stored in a structured format (dictionary) and later saved as a CSV file for further use by LIME


Second, run `SAC_Explain.py` , key points are here:

- **Model Initialization and Loading:** An instance of the Actor model is created and initialized with the appropriate parameters.

- **Separate Prediction Functions:** Define actor_predict_action_speed and actor_predict_action_steering_angle to return the specific action components.

- **Prepare Data for LIME:** Load and preprocess the data.


## Interpretation of Output

**Example output** 

For steering angle:

[('vx > 0.27', -0.03614449930860867), ('y > -1.00', -0.007705983263869293), ('vy > -0.14', 0.007525712199252107), ('x <= 1.00', 0.0)]


**Steering Angle Explanations**

1. *('vx > 0.27', -0.03614449930860867):* This means that when the x-velocity (vx) is greater than 0.27, it has a negative impact on the steering angle. The model tends to steer less (or more to the right) when the vehicle is moving faster in the x-direction.

2. *('y > -1.00', -0.007705983263869293):* When the y-position is greater than -1.00, it slightly decreases the steering angle. This might indicate that the model steers less (or more to the right) when the vehicle is not too far to the left of the road.

3. *('vy > -0.14', 0.007525712199252107):* When the y-velocity (vy) is greater than -0.14, it slightly increases the steering angle. This suggests that the model steers more to the left when the vehicle has a positive (or less negative) y-velocity.

4. *('x <= 1.00', 0.0):* The x-position being less than or equal to 1.00 has no impact on the steering decision for this instance.