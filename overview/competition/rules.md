Use of this competition
You cannot have multiple accounts.
Once you have deleted your user account, you will not be able to recreate it and participate in the same competition.
The maximum number of files a team can submit in a day is 5 times per team member. (e.g. 5 members = 25 files as a team per day) The limit resets at 0:00 AM (GMT).
It is prohibited to intentionally upload data that has not been analyzed (except for sample files) or has nothing to do with the competition.
Policy
During the competition, you must not privately share any source code or related data with any specific participants except for your team members. In addition, you must not share the source code you submit for final judging with any third party during the competition. If you share it, it will not be accepted as source code for the final submission.
Except for the cases described in the preceding paragraph, we do not restrict the public disclosure of algorithms, ideas, or other materials created in connection with your participation in this competition. However, sharing the data provided by Solafune Inc., including secondary works created from them, is prohibited. Also, when sharing, to indicate your participation in this competition, please include the following statement:
cf. @solafune(https://solafune.com) Use for any purpose other than participation in the competition or commercial use is prohibited. If you would like to use them for any of the above purposes, please contact us.

Please make sure that your article is available for all participants.
How to share your content
Use Github to share your content.
Post topic on discussion page
About datasets and weights
We prohibit the use of any external datasets in this competition.
The following models and weights are approved for use in the competitions.
You are allowed to use them as long as the trained models and/or weights used directly are included in the licenses.
Models and weights subject to CC (0 1.0 2.0 3.0 4.0), CC BY (1.0 2.0 3.0 4.0), MIT, BSD, U.S. Public Domain, and Apache 2.0 open source licenses.
*Available if the program, trained model and weights to be used directly are included in the available license.
We prohibit the use of programs, trained models, and weights that;
Cost money.
Only specific people are allowed to use them
Infringe the rights of any third party
Commercial use is not allowed
*We may add datasets, programs, trained models, and/or weights available other than as stated above at the discretion of the management.

*If you win a prize and are asked to submit the source code, you must clearly state the source and guarantee that it does not infringe the rights of any third party. Otherwise, the prize may be canceled.

About our contents
It is prohibited to release any data (including secondary products) provided by Solafune Inc.
You can publish your content (blogs, etc.) except for items that are prohibited by the management.
If a part of the winning source code, such as a generic function is made public, the management may ask you to stop releasing it. If you do not respond to the request, the prize may be cancelled.
Implementation
The tools used to train the models should be open-source and free (Python, R, etc.).
The source code that you submit when winning the prize should be divided into three: preprocessing module, learning module, and predicting module, as shown below, and the process should proceed if you run each one.
preprocessing: A module that reads the provided data, preprocesses the data, and outputs the file in a state ready for input to the model. Define separate preprocessing functions for training and evaluation. (preprocess_train.py, preprocess_test.py, etc.)
training: A module that reads the files created in the preprocessing stage and trains the model. Save the trained model, features, and Weight.
prediction: Module that reads the test data created in the preprocessing stage and the models created in the learning stage, and outputs the prediction results as a file. Output a file to be submitted.
It is recommended that the winning users contain their implementation in a Containerized Environment to harmonize users’ development environment with our code inspection environment.
You may use DockerFile to provide us with the environment that you used.
Please use a CUDA 11.8 or above environment if you use NVIDIA’s GPUs.
About Solafune-Tools
Solafune-Tools is an open-source software project designed to empower the remote sensing community by providing essential tools for research and geospatial analysis. It facilitates remote-sensing-related tasks and enhances user experience in Solafune-hosted competitions, enabling seamless data processing, analysis, and collaboration.
Check this link for guidelines

About Team
The maximum number of team members is five.
When your team wins prizes, all team members need to verify their identity.
The distribution of prize money in the team should be reported by the deadline of the source code acceptance procedure.
Overall ranking
This competition is subject to the overall ranking. Results will be reflected after the competition is completed.

Requests from Solafune
We expect competitors to participate in the competition with a practical approach to solving social or corporate issues or sharing their research results.
To ensure fairness and prevent fraud, the competition may add or change the rules without notice. Also, we expect to receive opinions from the competitors through interviews and questionnaires. Please understand them for the improvements of the service.