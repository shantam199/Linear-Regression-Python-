# E-Commerce Regression

This is the code for a linear regression model used to predict revenue amount based on product usage.

## Overview

I built a simple linear regression model using the [Scikit-learn](http://scikit-learn.org/stable/) library. The inputs and outputs were both numeric values.

## Dependencies

- numpy
- pandas
- scikit-learn

Install dependencies using [pip](https://pip.pypa.io/en/stable/).

## Dataset

The dataset was taken from the [Udemy Python for DS and ML bootcamp](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/). The particular subset I used (/input/customers.csv) contains 500 observations (customers) and 7 attributes.

| Column  | Definition |
| ------------- | ------------- |
| Email  | Customer's email ID  |
| Address  | Customer's address  |
| Avatar  | Colour of customer's profile  |
| Avg. Session Length  | Average in-person consulting time spent by customer  |
| Time on App  | Time spent by customer on mobile app  |
| Time on Website  | Time spent by customer on website  |
| Length of Membership  | For how many years has the customer been with the company?  |
| Yearly Amount Spent  | Average amount spent by customer on company app/website  |

## Usage

Run the notebook on a localhost server using `jupyter notebook`.
