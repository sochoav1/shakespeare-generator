import axios from 'axios';
import React, { useState } from 'react';
import './App.css';
import GenerateButton from './components/GenerateButton';
import GeneratedText from './components/GeneratedText';
import TextInput from './components/TextInput';

const App = () => {
  const [maxTokens, setMaxTokens] = useState(100);
  const [generatedText, setGeneratedText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleGenerateText = async () => {
    setLoading(true);
    setGeneratedText('');
    try {
      const response = await axios.post('http://localhost:8000/generate', {
        max_tokens: maxTokens,
      });
      setGeneratedText(response.data.generated_text);
    } catch (error) {
      console.error('Error generating text:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Shakespeare Text Generator</h1>
      <TextInput value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} />
      <GenerateButton onClick={handleGenerateText} loading={loading} />
      <GeneratedText text={generatedText} />
    </div>
  );
};

export default App;
