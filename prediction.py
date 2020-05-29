from preprocessing import *
from validation import *
from model import *

# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)