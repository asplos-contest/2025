# 

# **The ASPLOS 2025 / EuroSys 2025 Contest on an Optimized Neuron Kernel Interface (NKI) Implementation of Llama 3.2 1B (Inference)**

Teams from around the globe are invited to contribute submissions toward producing the fastest implementation of the Llama 3.2 1B model (inference only), written using the Neuron Kernel Interface (NKI) programming interface and running on Amazon ML hardware (Trainium/Inferentia). Prizes will be awarded to top-ranking teams who commit to open-sourcing their solutions prior to next year's conference.

|  | Rank | Prize |  |
| :---- | ----: | ----: | :---- |
|  | First Place 🥇 | $25,000 |  |
|  | Second Place 🥈 | $10,000 |  |
|  | Third Place 🥉 | $5,000 |  |

<p align="center">
<a href="https://forms.gle/k56AmQ764t3ai3Gq7">
<img src="images/register.png" width="200">
</a>
</p>

# Important Dates 

|  | Date | Event |  |
| :---- | :---: | ----- | :---- |
|  | ~~2024-12-01~~ | Contest [Announced](https://www.sigarch.org/call-participation/the-asplos-2025-eurosys-2025-contest-track/) |  |
|  | ~~2025-01-18~~ | [Contest GitHub Repository & Benchmark Subset Released](https://github.com/aws-samples/nki-llama)  |  |
|  | ~~2025-02-03~~ | Application Deadline for [Student Travel Grants](https://www.asplos-conference.org/asplos2025/student-travel-grants/) |  |
|  | 2025-02-22 | Contest Registrations & Preliminary Submissions Due\* |  |
|  | 2025-03-01 | Contest Final Submissions Due\* |  |
|  | 2025-03-03 | [Early Registration Deadline for ASPLOS 2025 / EuroSys 2025](https://www.asplos-conference.org/asplos2025/registration/) |  |
|  | 2025-03-30 | Contest Special Session during [ASPLOS 2025 / EuroSys 2025 Workshops](https://www.asplos-conference.org/asplos2025/workshops-and-tutorials/) |  |
|  | 2025-04-01 | Contest Winners Announced during ASPLOS 2025 / EuroSys 2025 Conference |  |

\*Submissions are due by 11:59pm at any time on Earth.

# Problem Description

Amazon Web Services has two family of machine learning chips, called
Trainium and Inferentia. AWS Neuron SDK is an SDK with a compiler and
profiling tools for programming these devices using high-level
libraries like PyTorch. AWS [recently released a new programming
interface called Neuron Kernel Interface
(NKI)](https://aws.amazon.com/about-aws/whats-new/2024/09/aws-neuron-nki-nxd-training-jax/)
that gives programmers down-to-the-metal access to Trainium/Inferentia
hardware features, potentially unlocking even greater performance
opportunities.

For this contest, teams will submit code that leverages NKI to
implement the Llama3.2 1B model, targeting a single Trainium1 (trn1)
chip.

# Contest GitHub Repository and Benchmark Subset

Please consult the following GitHub repository for full contest details:
[https://github.com/aws-samples/nki-llama](https://github.com/aws-samples/nki-llama).

# Contest Organizers

* Emery Berger (Amazon Web Services), [emerydb@amazon.com](mailto:emerydb@amazon.com)
* Aninda Manocha (Amazon Web Services)
* Wei Tang (Amazon Web Services)
* Emily Webber (Amazon Web Services)
* Ziyang Xu (Amazon Web Services)
 
# Contest Sponsor

![Cloud Computing Services - Amazon Web Services (AWS)](https://github.com/asplos-contest/2025/blob/main/images/aws-logo.jpg)

Amazon Web Services

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>
