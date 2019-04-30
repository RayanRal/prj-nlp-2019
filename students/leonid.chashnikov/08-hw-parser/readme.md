
#### LogReg, 1st version

First version, only top queue/stack, no feats
LogisticRegression(random_state=42, solver="sag", multi_class="multinomial", max_iter=1000)

              precision    recall  f1-score   support

        left       0.81      0.84      0.83      7936
      reduce       0.64      0.56      0.60      5916
       right       0.71      0.73      0.72      7686
       shift       0.82      0.84      0.83      8088

   micro avg       0.76      0.76      0.76     29626
   macro avg       0.74      0.74      0.74     29626
weighted avg       0.75      0.76      0.75     29626

Total: 15774
Correctly defined: 10443
UAS: 0.66


#### LogReg, 2nd version

After adding stack 2 POS, queue 2-3-4 POS, and feats for stack/queue 1
LogisticRegression(random_state=42, solver="sag", multi_class="multinomial", max_iter=1000)

              precision    recall  f1-score   support

        left       0.87      0.88      0.88      7936
      reduce       0.77      0.69      0.73      5916
       right       0.78      0.82      0.80      7686
       shift       0.86      0.88      0.87      8088

   micro avg       0.82      0.82      0.82     29626
   macro avg       0.82      0.82      0.82     29626
weighted avg       0.82      0.82      0.82     29626

Total: 15774
Correctly defined: 11341
UAS: 0.72


#### DecisionTreeClassifier

3 exceptions 'list index out of range' on UAS scoring (dep_parse), that negatively influenced UAS,
but it's still better than LogReg.

              precision    recall  f1-score   support

        left       0.90      0.92      0.91      7936
      reduce       0.78      0.74      0.76      5916
       right       0.83      0.83      0.83      7686
       shift       0.90      0.91      0.90      8088

   micro avg       0.86      0.86      0.86     29626
   macro avg       0.85      0.85      0.85     29626
weighted avg       0.86      0.86      0.86     29626

Total: 15774
Correctly defined: 11979
UAS: 0.76


#### RandomForestClassifier

20 exceptions on dependency_parse for UAS scoring. And overall result slightly worse than Decision Tree

              precision    recall  f1-score   support

        left       0.83      0.94      0.88      7936
      reduce       0.80      0.64      0.71      5916
       right       0.81      0.82      0.81      7686
       shift       0.88      0.88      0.88      8088

   micro avg       0.83      0.83      0.83     29626
   macro avg       0.83      0.82      0.82     29626
weighted avg       0.83      0.83      0.83     29626

Total: 15774
Correctly defined: 11838
UAS: 0.75

