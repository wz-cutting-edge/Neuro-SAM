from flask import Flask, request, jsonify
from flask_cors import CORS
from demo import HybridSentimentModel, tokenizer, lda_model, vectorizer, adapt_topic_dist
import torch

app = Flask(__name__)
CORS(app)

model = HybridSentimentModel.from_pretrained("aiguy68/neurosam-model")
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Text processing
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Topic distribution processing
        dtm = vectorizer.transform([text])
        topic_dist_raw = lda_model.transform(dtm)
        
        # Create adapted topic distributions
        topic_dist = adapt_topic_dist(topic_dist_raw, 15)
        topic_dist_25 = adapt_topic_dist(topic_dist_raw, 25)

        # Model inference
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                topic_dist=topic_dist,
                topic_dist_25=topic_dist_25
            )
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        sentiment_labels = ["Negative", "Neutral", "Positive"]
        confidence_scores = {
            label: round(probabilities[0][idx].item() * 100, 2)
            for idx, label in enumerate(sentiment_labels)
        }

        return jsonify({
            "sentiment": sentiment_labels[prediction],
            "confidence_scores": confidence_scores
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
