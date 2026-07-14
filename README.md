# Random Forest Regression

## What is Random Forest Regression?
Random Forest Regression is an ensemble learning method that builds multiple decision trees and merges their predictions together to get a highly accurate and stable numerical estimate.

## How It Works:
Bootstrap Sampling: It creates random subsets of your dataset (with replacement) to train each individual tree.
Feature Bagging: At each split in a tree, it only considers a random subset of features. This ensures the trees are diverse and uncorrelated.
Averaging: Every tree makes its own prediction, and the final output is the mathematical average of all the trees:
 
## When to Use It:
When your data has complex, non-linear relationships or intricate feature interactions.
When you have high-dimensional data (lots of features).
When your dataset has missing values or outliers.
As a strong, "out-of-the-box" benchmark model.

## Pros & Cons:
### Advantages	
- No Overfitting: Averaging many trees prevents overfitting.	
- Robust: Handles missing data and outliers incredibly well.	
- Feature Importance: Tells you which features matter most.	
### Limitations:
- No Extrapolation: Cannot predict values outside the range of training data.
- Slow & Heavy: Can be slow to train and uses significant memory.
- Black Box: Hard to interpret compared to a single decision tree.


## Result:
