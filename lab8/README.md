Perceptronul și rețele de perceptroni în Scikit-learn

Exerciții
1. Antrenati un perceptron pe multimea de puncte 3d, pana cand eroare nu se
imbunatateste cu 1e-5 fata de epocile anterioare, cu rata de invatare 0.1. Calculati
acuratetea pe multimea de antrenare si testare, apoi afisati ponderile, bias-ul si
numarul de epoci parcuse pana la convergenta. Plotati planul de decizie al
clasificatorului cu ajutorului functiei plot3d_data_and_decision_function.

2. Antrenati o retea de perceptroni care sa clasifice cifrele scrise de mana
MNIST. Datele trebuie normalizate prin scaderea mediei si impartirea la deviatia
standard. Antrenati si testati urmatoarele configuratii de retele:
a. Functia de activare ‘tanh’, hidden_layer_sizes=(1),
learning_rate_init=0.01, momentum=0 (nu vom folosi momentum),
max_iter=200 (default)
b. Functia de activare ‘tanh’, hidden_layer_sizes=(10),
learning_rate_init=0.01, momentum=0 (nu vom folosi momentum),
max_iter=200 (default)
c. Functia de activare ‘tanh’, hidden_layer_sizes=(10),
learning_rate_init=0.00001, momentum=0 (nu vom folosi momentum),
max_iter=200 (default)
d. Functia de activare ‘tanh’, hidden_layer_sizes=(10),
learning_rate_init=10, momentum=0 (nu vom folosi momentum),
max_iter=200 (default)
e. Functia de activare ‘tanh’, hidden_layer_sizes=(10),
learning_rate_init=0.01, momentum=0 (nu vom folosi momentum),
max_iters=20
f. Functia de activare ‘tanh’, hidden_layer_sizes=(10, 10),
learning_rate_init=0.01, momentum=0 (nu vom folosi momentum),
max_iter=2000
g. Functia de activare ‘relu’, hidden_layer_sizes=(10, 10),
learning_rate_init=0.01, momentum=0 (nu vom folosi momentum),
max_iter=2000
h. Functia de activare ‘relu’, hidden_layer_sizes=(100, 100),
learning_rate_init=0.01, momentum=0 (nu vom folosi momentum),
max_iter=2000
i. Functia de activare ‘relu’, hidden_layer_sizes=(100, 100),
learning_rate_init=0.01, momentum=0.9, max_iter=2000
j. Functia de activare ‘relu’, hidden_layer_sizes=(100, 100),
learning_rate_init=0.01, momentum=0.9, max_iter=2000, alpha=0.005)
