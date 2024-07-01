import React from "react";

const TextInput = ({ value, onChange }) => (
    <div>
        <label>
            Maximum Tokens:
        <input type="number" value={value} onChange={onChange} />
        </label>
    </div>
);

export default TextInput;
