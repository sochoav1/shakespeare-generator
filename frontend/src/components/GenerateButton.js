import React from 'react';

const GenerateButton = ({ onClick, loading }) => (
    <button onClick={onClick} disabled={loading}>
        {loading ? 'Generating...' : 'Generate'}
    </button>
);

export default GenerateButton;
