# Fast and Low-resource semi-supervised Abdominal oRgan sEgmentation in CT （FLARE 2022）

Challenge link: https://flare22.grand-challenge.org/

## Overview

The FLARE 2022 challenge has three main features:

- *Task*: we use a semi-supervised setting that focuses on how to use unlabeled data.
- *Dataset*:  we curate a large-scale and diverse abdomen CT dataset, including 2300 CT scans from 20+ medical groups.
- *Evaluation measures*: we not only focus on segmentation accuracy but also on segmentation efficiency and resource consumption.


## How to participate
### **Stage 0**. Get the qualification
- The challenge submission is based on Docker container. So, participants should demonstrate basic segmentation skills and the ability to encapsulate their methods in Docker. We provide a playground for participants to practice. Participants should 

- Develop any segmentation method (e.g., U-Net) based on the playground training dast and encapsulate the method by Docker.
- Use Docker to predict the testing set and record the predicting process as a video (mp4 format).
- Submit the segmentation results here and upload your Docker to DockerHub. Send the (1) docker hub link, (2) download link to the recorded inference mp4 video, and (3) the screenshot of your playground leaderboard results (Mean DSC>0.3) to MICCAI.FLARE@aliyun.com.
- After reviewing your submission, we will get back to you with an Entry Number, then you can join FLARE22 Challenge.
If you have won an award in MICCAI FLARE21 Challenge, this step can be exempt and you can directly go to Stage 1. Your Entry Number is the Certificate Number in your award certificate.

- If you have made a successful submission in other Docker-based MICCAI Challenges (e.g., KiTS 2021, BraTS 2021...), you can also be exempt from Stage 0. Please send supporting materials to MICCAI.FLARE@aliyun.com, we will get back to you with an Entry Number.

- Each team only needs one Entry Number.  

### **Stage 1**. Join the challenge and download data
- Click the green 'Join' button to participate in the FLARE22 Challenge and fill out the online registration form (GoogleDoc or TencentDoc).
- Send the signed Challenge Rule Agreement Consent form (GoogleDrive, Tencent Netdisk) to MICCAI.FLARE@aliyun.com via your affiliation E-mail (Gmail, 163.com, qq.com, outlook.com et al. will be ignored without notice). Email Subject: Signed FLARE Rule Agreement_EntryNumber_grand challenge user name
- We will approve your participation request in 1-2 days and you can download the data on the Dataset page.
- ***Note***: Please only use lowercase letters and numbers in your team name! No spacing or special characters.

### **Stage 2**.  Develop your model and make validation submissions
- We offer three official submission opportunities on the validation set. To avoid submission jams, the three opportunities are assigned in three months (See Important Dates).
- To make official submissions, please send us (MICCAI.FLARE@aliyun.com) a download link to your Docker container (teamname.tar.gz) and a methodology paper (following the template).  When the evaluation is finished, we will return back all metrics via email.
- Meanwhile, all teams have one chance per day to get the DSC score on the validation set by directly submitting the segmentation results on the Submit page. (All participants must submit in a form of teams, including the teams with a single user. Personal submission is not allowed.) Teams that violate this rule (e.g, making multiple submissions in a day via different team members' accounts) will be banned from submitting for a week!
### **Stage 3**. Make testing submissions
- To avoid overfitting the testing set, we only offer one successful submission opportunity on the testing set.
- The submission should include a Docker container (teamname.tar.gz) and a methodology paper.
- The submitted Docker container will be evaluated with the following commands. If the Docker container does not work or the paper does not include all the necessary information to reproduce the method, we will return back the error information and review comments to participants.

## Important Dates

- **15 March 2022 (12:00 AM EST)**: Launch of challenge and release of training data.
- **31 March 2022 (12:00 AM EST)**: Release of validation data. Docker and short paper submission of validation set opening.
- **15 April 2022 (12:00 AM EST)**: Deadline for the first validation submission.
- **15 May 2022 (12:00 AM EST)**: Deadline for the second validation submission.
- **15 June 2022 (12:00 AM EST)**: Deadline for the third validation submission and new registration. Docker and short paper submission of testing set opening.
- **15 July 2022 (12:00 AM EST)**: Deadline for testing submission.
- **15 August 2022 (12:00 AM EST)**: Invite top teams to prepare presentations and participate in MICCAI22 Satellite Event.
18/22 September 2022: Announce final results.


## Ranking Scheme
To keep the balance between accuracy metrics and efficiency metrics in the ranking scheme, we assign half weight to CPU and GPU metrics. The ranking scheme includes the following three steps:

- **Step 1**. Compute the five metrics for each testing case.
- **Step 2**. Rank participants for each of the 200 testing cases and each metric; Thus, each participant will have 200x4 rankings.
- **Step 3**. Average all these rankings (GPU and CPU metrics have half weights).
ChallengeR will be used to analyze and rank the results. https://github.com/wiesenfa/challengeR