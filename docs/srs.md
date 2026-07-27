# Software Requirements Specification (SRS)

# 1. Table of Contents

- [2 Introduction](#2-introduction)
  - [2.1 Scope](#21-scope)
  - [2.2 Definitions, Acronyms, and Abbreviations](#22-definitions-acronyms-and-abbreviations)
- [3 Overall description](#3-overall-description)
  - [3.1 Product perspective](#31-product-perspective)
  - [3.2 Interfaces](#32-interfaces)
  - [3.3 Product functions](#33-product-functions)
  - [3.4 User characteristics](#34-user-characteristics)
  - [3.5 Constraints](#35-constraints)
  - [3.6 Skills availability](#36-skills-availability)
  - [3.7 Time-Frame](#37-time-frame)
  - [3.8 Budget](#38-budget)
- [4 Specific requirements](#4-specific-requirements)
  - [4.1 External Interface Requirements](#41-external-interface-requirements)
    - [4.1.1 User Interfaces](#411-user-interfaces)
    - [4.1.2 Hardware Interfaces](#412-hardware-interfaces)
    - [4.1.3 Software Interfaces](#413-software-interfaces)
    - [4.1.4 Communications Interfaces](#414-communications-interfaces)
  - [4.2 Functional Requirements](#42-functional-requirements)
    - [4.2.1 Feature 1: ](#421-feature-1)
    - [4.2.2 Feature 2: ](#422-feature-2)
    - [4.2.3 Feature #3: Component lifetime prediction](#423-feature-3-component-lifetime-prediction)
  - [4.3 Use Cases](#43-use-cases)
    - [4.3.1 Use Case #1](#431-use-case-1)
    - [4.3.2 Use Case #2](#431-use-case-2)
  - [4.4 Non-Functional Requirements](#44-non--functional-requirements)
    - [4.4.1 Performance](#441-performance)
    - [4.4.2 Scalability](#442-scalability)
    - [4.4.3 Portability](#443-prtability)
    - [4.4.4 Compatibility](#444-compatibility)
    - [4.4.5 Reliability](#445-reliability)
    - [4.4.6 Maintainability](#446-maintainability)
    - [4.4.7 Availability](#447-availability)
    - [4.4.8 Security](#448-security)
    - [4.4.9 Usability](#449-security)
  - [4.5 Design Constraints](#45-design-constraints)
  - [4.6 Logical Database Requirements](#46-logical-database-requirements)
  - [4.7 Other Requirements](#47-other-requirements)

#

## <font color="gray"> The software requirement specification (SRS) document is a living document that can be modified or extended throughout the entire project development process. It should serve as a reference for designing, implementing, and testing the software tool. </font>


## <font color="red"> Don’t get stressed if you cannot provide input to every item from the beginning.</font>


## 2. Introduction

### 2.1 Scope

<font color="gray">*This subsection should:*

*a) Identify the software product(s) to be produced by name (e.g., INARA, SEGROcalc.);*

*b) Explain what the software product(s) will, and, if necessary, will not do;*

*c) Describe the application of the software, including relevant benefits, objectives, and*</font>

<font color="gray">*goals*</font>

### 2.2 Definitions, Acronyms, and Abbreviations

<font color="gray"> *Provide the definitions of all terms, acronyms, and abbreviations required to properly interpret the SRS.* </font>


## 3 Overall description

<font color="gray"> *This section should describe the general factors that affect the product and its requirements. This section does not state specific requirements, but rather a general background.* </font>

### 3.1 Product perspective

<font color="gray"> *Put the software into perspective with other related products. If the software is independent and totally self-contained. If it would be developed from scratch.*

*If the software product is part of a larger system, explain how the system's requirements relate to the software's functionality and identify any interfaces between the software and the system. A block diagram showing the major components of the larger system, interconnections, and external interfaces can be helpful.* </font>

### 3.2 Interfaces

<font color="gray"> *If applicable list:*

- *System interfaces*
- *User interfaces*
- *Hardware interfaces*
- *Software interfaces*
- *Communications interfaces* </font>

### 3.3 Product functions

 <font color="gray"> *Use graphical or textual methods to show the major functions of the software and their relationship, without going into details.* </font>

### 3.4 User characteristics

 <font color="gray"> *Describe the intended users of the product. (e.g. MCL researchers, external researchers, industrial partners, commercial use.)* </font>

### 3.5 Constraints

<font color="gray"> *List any constrain the software developer and designer should consider.  (e.g. Regulatory policies, Licences, Hardware limitations, Memory, Interfaces to other applications; Parallel operation, Safety and security considerations, Development time, Budget)* </font>

### 3.6 Skills availability

<font color="gray"> *Identify whether we have the necessary knowledge in-house/project consortium, whether we will develop the knowledge during the project., or whether we will have to outsource some part(s) of the project.* </font>

### 3.7 Time-Frame

<font color="gray"> *Specify the time frame for the software development.* </font>

### 3.8 Budget

<font color="gray"> *Specify the budget for software development, including resources for support (software development team).*

## 4 Specific requirements

<font color="gray"> *Specify all software requirements with sufficient detail to enable designers to create a system to meet those requirements, and testers to verify them.* </font>

### 4.1 External Interface Requirements

#### 4.1.1 User Interfaces

#### 4.1.2 Hardware Interfaces

Server for web interface/application?

#### 4.1.3 Software Interfaces


#### 4.1.4 Communications Interfaces


### 4.2 Functional Requirements

<font color="gray"> *Describes specific features of the software* </font>

#### 4.2.1 Feature 1: 

  **a. Description:** 
  
  **b. Inputs:**

  **c. Processing:**

  **d. Outputs:**

  **e. Remarks:**

  **f. Error Handling:**

#### 4.2.2 Feature 2: 

  **a. Description:**   

  **b. Inputs:**

  **c. Processing:**

  **d. Outputs:**

  **e. Remarks:**

  **f. Error Handling:** 

#### 4.2.3 Feature 3: 

  **a. Description:** 

  **b. Inputs:**

  **c. Processing:**

  **d. Outputs:**

  **e. Remarks:**

  **f. Error Handling:** 

### 4.3 Use Cases

<font color="gray"> *Specify use cases to guide and verify software requirements. During the software implementation process, it is possible to define use cases as needed and without prior planning.* </font>

#### 4.3.1 Use Case 1

<font color="gray">*Example*

- ***Need:** In the GUI, upload data to the database*
- ***Remarks:***
  - *Data are available as 2-column csv or raw data (txt/ASCII file): x … Raman shift (frequency), y … Intensity*
  - *The user should have the capability to provide metadata with the spectra (by hand in the GUI; e.g. the units)*
  - *Default unit for demo version:*
    - <i>Raman shift in cm<sup>-1</sup></i>
    - *Arbitrary units, not normalized*
- ***Workflow:***
  - *Select data type: Experimental or theoretical*  
  - *Select a data file (csv or raw ASCII data in 2-column format).*
  - *Preview the (raw) dataset (without units).*  
  - *Add metadata;*
  - *Save data on the database*</font>

#### 4.3.2 Use Case 2

…

### 4.4 Non-Functional Requirements

<font color="gray"> *If applicable, describe non-functional requirements related to the following attributes [[2](#references)]:* </font>

#### 4.4.1 Performance

<font color="gray"> *How fast the system (or its component) should responds to certain users’ actions under a certain workload.*

*(e.g. Product search results should be displayed within 3 seconds for 90% of the searches under normal conditions.)* </font>

#### 4.4.2 Scalability

<font color="gray"> *Assesses the highest workloads under which the system will still meet performance and usability requirements.*

*(e.g. The website must be scalable enough to support 1,000,000 visits at the same time while maintaining optimal performance.)* </font>

#### 4.4.3 Portability

<font color="gray"> *Determine if the system or its elements can work in different environments.*

*(e.g. The software should run within a Docker container. The web application needs to work consistently on all major web browsers such as Chrome, Firefox, Safari, and Edge.)* </font>

#### 4.4.4 Compatibility

<font color="gray"> *Defines how the system can coexist and interact with another system in the same environment.*

*(The software for visualizing crystalline structures must support common formats for importing and exporting atomic coordinates, including CIF, xyz, xsf, cfg, and POSCAR).* </font>

#### 4.4.5 Reliability

<font color="gray"> *The likelihood of the software meeting performance standards and generating accurate output within a specified time*

*(e.g. The system must perform without failure in 95 percent of use cases during a month. The system should be able to handle and recover from errors without data loss or incorrect data processing.)*</font>


#### 4.4.6 Maintainability

<font color="gray"> *Defines the time required to fix or change a solution or its component to improve performance or adapt to a changing environment.*

*(e.g.* *The video streaming service should automatically adjust the video quality based on the user's internet speed to avoid buffering.)* </font>


#### 4.4.7 Availability

<font color="gray"> *Describes how likely the system is accessible to a user at a given point in time.*

*(e.g.  how long can be unavailable without impacting operations.)* </font>


#### 4.4.8 Security

<font color="gray"> *Assess how data inside the system will be protected against malware attacks, data breaches, data loss, or unauthorized access.*  

*(e.g.  data access should be granted by permissions)* </font>


#### 4.4.9 Usability

<font color="gray"> *Indicates how effectively and easy easily users can learn and use the system*

*(e.g. Users should be able to find desired products within three clicks from the homepage.)* </font>


### 4.5 Design Constraints

<font color="gray"> *Specify design constrains imposed by other standards, company policies, hardware limitation, etc. that will impact this software project.* </font>

### 4.6 Logical Database Requirements

<font color="gray"> *If databases are used, specify what logical requirements exist for data formats (i.e., data entities and their relationships), storage capabilities, data retention, data integrity, etc.* </font>


### 4.7 Other Requirements

<font color="gray"> *Provide details of any additional requirement that has not been covered in the previous sections.* </font>


### References

[1]: IEEE Std 1233, 1998 Edition , vol., no., pp.1-36, 29 Dec. 1998, doi: 10.1109/IEEESTD.1998.88826

[2]: Examples and definitions  were taken from: [Nonfunctional Requirements (NFR):Examples, Types, Approaches (altexsoft.com)](https://www.altexsoft.com/blog/non-functional-requirements/).
