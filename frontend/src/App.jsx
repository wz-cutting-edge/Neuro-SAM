import React, { useState } from 'react';
import './App.css';

function App() {
    {/* Variables to store Input Text, Results, and Loading state */}
    const [inputText, setInputText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    {/* 
        Function to handle submission of Input Text
        The input text is trimmed to remove any leading or trailing whitespace.
        The loading state is used to disable the input and button while the request is being processed. (True = disabled)
        It sends a POST request to the Flask backend with the input text and updates the result state with the response.
        The response contains the predicted sentiment and confidence scores.
     */}
    const handleSubmit = async () => {
        if (!inputText.trim()) return;
    
        setLoading(true);
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: inputText }),
                });
            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error:', error);
            setResult({ error: "Failed to get prediction" });
        }
        setLoading(false);
    };

    {/* Start of App Structure */}
    return (
        <>
            {/* Main Container */}
            <div className="app">

                {/* Header Section */}
                <header className="app-header">
                    <img src="/neuro-sam.png" />
                    <h1>Neuro-SAM: Text Sentiment Classifier</h1>
                </header>

                {/* Main Section */}
                <main className="app-main">

                    {/* Form Container */}
                    <div className="input-box">
                        <label htmlFor="text-input">Enter Text to Classify:</label>
                        <textarea
                            id="text-input"
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            rows="4"
                            disabled={loading}
                            defaultValue="Enter it here..."
                        />
                        <button 
                            onClick={handleSubmit}
                            disabled={loading || !inputText.trim()}
                        >
                            {loading ? 'Analyzing...' : 'Analyze Sentiment'}
                        </button>
                    </div>

                    {result && (
                        /* Result Container */
                        <div className="result-box">
                            {result.error ? (
                                <p className="error">Error: {result.error}</p>
                            ) : (
                                <>
                                    <h2>Analysis Result</h2>
                                    <p>Predicted Sentiment: <strong>{result.sentiment.charAt(0).toUpperCase() + result.sentiment.slice(1)}</strong></p>
                                    <div className="confidence-scores">
                                        <h3>Confidence Scores:</h3>
                                        <ul>
                                            {Object.entries(result.confidence_scores).map(([label, score]) => (
                                                <li key={label}>
                                                    {label.charAt(0).toUpperCase() + label.slice(1)}: 
                                                    <strong> {score}%</strong>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </main>

                {/* Footer Section */}
                <footer className="app-footer">
                    &copy; 2025 Neuro-SAM. All rights reserved.
                </footer>
            </div>
        </>
    );
}

export default App;
