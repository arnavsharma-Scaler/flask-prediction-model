from flask import Flask, request, render_template, jsonify
from model import EmbeddingClassi
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
model = EmbeddingClassi()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    form_values = list(request.form.values())
    print(form_values)

    if form_values:
        query_text = next(iter(form_values))
        print(query_text)

        prediction = model.query(query_text)
        knn = NearestNeighbors(n_neighbors=2)
        knn.fit(model.pca_embeddings)
        x = prediction
        neighbours = knn.kneighbors(x, return_distance = False)
        df1 = model.df.copy().reset_index()
        df1['labels'] = model.labels

        for i in neighbours:
            dt = df1.loc[i,['Ticket Subject','Ticket Description','Resolution','labels']]
        output = dt
        
        return render_template('index.html', prediction_text='Output is {}'.format(output))
    else:
        return render_template('index.html', prediction_text='No input provided')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    if 'query' in data:
        prediction = model.query(data['query'])

        knn = NearestNeighbors(n_neighbors=2)
        knn.fit(model.pca_embeddings)
        x = prediction
        neighbours = knn.kneighbors(x, return_distance = False)
        df1 = model.df.copy().reset_index()
        df1['labels'] = model.labels

        for i in neighbours:
            dt = df1.loc[i,['Ticket Subject','Ticket Description','Resolution','labels']]
        print(dt)
        # Convert DataFrame to dictionary
        output = dt.to_dict()
        return jsonify(output)
    else:
        return jsonify({"error": "Missing 'query' in request data"}), 400
    
if __name__ == "__main__":
    app.run(debug=True)