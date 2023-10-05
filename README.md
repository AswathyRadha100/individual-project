--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                                                                                                                                                           #
# Walmart's Goldmine of Predictions
#                                           
#                                                                                                                                                                                                           #  
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Description** 

###  This project emphasizes the value of data-driven decision-making in the retail industry and highlights the potential of time series Prophet model, to enhance retail sales forecast.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Goal** 

###  The main goals of the project are
  - &#9733; Construct an ML model based on time series for retail sales forecast
  - &#9733; Identify key drivers of sales forecast
  - &#9733; Enhance retail prediction accuracy
  - &#9733; Provide insights into retail prediction variations
  - &#9733; Deliver a write up for my professional portfolio 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Initial Thoughts**

### In the initial phases, our primary objectives revolve around data exploration, identifying the factors affecting sales forecast, and constructing time series Prophet model to enhance the accuracy of retail prediction. This foundational stage lays the groundwork for harnessing data effectively in predicting sales in retail.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **The Plan**
- &#9733; Acquire data from https://data.world/ahmedmnif150/walmart-retail-dataset/workspace/file?filename=Walmart-Retail-Dataset.csv
- &#9733; Prepare retail data for analysis:
  -  Data Cleaning and Feature Engineering for Retail Sales:
     - &#9642; Drop rows and columns with high percentages of missing values.
     - &#9642; Impute missing values    
     - &#9642; Data Type Conversion
- &#9733; Explore data to predict the sales:
  -  Answer important questions
     - &#9642; What is the overall trend in sales over time?
     - &#9642; Can we forecast future sales based on historical data?
     - &#9642; How do sales patterns vary across different customer segments over time?
     - &#9642; Is there a relationship between the order quantity and the order date? Are there any specific trends or patterns in order quantities over time?
- &#9733; Model Selection:
  -   Time series forecasting models
     - &#9642; Prophet
     - &#9642; ARIMA
     - &#9642; SARIMA
     
- &#9733; Data Splitting and Model Training:
  -  Divide the dataset into train,and test sets
     - &#9642; Train chosen models on training dataset
- &#9733; Model Evaluation:
  -   Check the performance of models on the test dataset
  - Metrics used
     - &#9642; Mean Absolute Error (MAE)
     - &#9642; Mean Squared Error (MSE)
   

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Data Dictionary** 



| Column Name          | Non-Null Count | Data Type         | Definition                                     |
|----------------------|----------------|-------------------|-----------------------------------------------|
| city                 | 978,393        | object            | City name where the order was placed          |
| customer_age         | 978,393        | int64             | Age of the customer                           |
| customer_name        | 978,393        | object            | Name of the customer                          |
| customer_segment     | 978,393        | object            | Customer segment or category                  |
| discount             | 978,393        | float64           | Discount applied to the order                 |
| order_date           | 978,393        | datetime64[ns]    | Date when the order was placed                |
| order_id             | 978,393        | object            | Unique identifier for the order               |
| order_priority       | 978,393        | category          | Priority of the order (e.g., high, medium)    |
| order_quantity       | 978,393        | int64             | Quantity of products in the order             |
| product_base_margin  | 978,393        | float64           | Profit margin of the product                  |
| product_category     | 978,393        | category          | Category of the product (e.g., electronics)   |
| product_container    | 978,393        | category          | Type of container for shipping                |
| product_name         | 978,393        | category          | Name of the product                           |
| product_sub_category | 978,393        | category          | Sub-category of the product                   |
| profit               | 978,393        | float64           | Profit from the order                         |
| region               | 978,393        | category          | Region where the order was placed             |
| sales                | 978,393        | float64           | Total sales amount                            |
| ship_date            | 978,393        | datetime64[ns]    | Date when the order was shipped               |
| ship_mode            | 978,393        | category          | Shipping mode (e.g., express, standard)       |
| shipping_cost        | 978,393        | float64           | Cost of shipping the order                    |
| state                | 978,393        | category          | State or region where the order was placed    |
| unit_price           | 978,393        | float64           | Price per unit of the product                |
| zip_code             | 978,393        | category          | ZIP code associated with the order            |


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Steps to Reproduce** 

## Ordered List:
     1. Clone this repo.
     2. Acquire the data from https://data.world/ahmedmnif150/walmart-retail-dataset/workspace/file?filename=Walmart-Retail-Dataset.csv
     3. Run data preprocessing and feature engineering scripts.
     4. Explore data using provided notebooks.
     5. Train and evaluate time series models using Colab to run Prophet and uploaded the csv files from colab in jupyter notebook and explored the results in jupyter notebook. 
     6. Code used for Prophet model in Colab is commented out in the jupyter notebook due to version conflict (can't run the code in jupyter notebook)
     6. Replicate the retail sales forecast process using the provided instructions.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Recommendations**

## Actionable recommendations based on project's insights:
- &#9733; Dynamic Inventory Management
- &#9733; Marketing and Promotions
- &#9733; Collaborative Planning
- &#9733; Continuous Monitoring

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Takeaways and Conclusions**
In conclusion, this project has unveiled essential insights into historical sales patterns, customer segmentation, year-end sales surges, and inventory management. These findings, along with the successful implementation of the Prophet model, empower us to make data-driven decisions, optimize inventory management, and strategically plan marketing campaigns. This project equips us to enhance supply chain operations, drive meaningful improvements in future projects, and underscores our commitment to harnessing data for informed decision-making and achieving tangible business enhancements

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


