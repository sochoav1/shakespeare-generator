import React from 'react';
import { TypeAnimation } from 'react-type-animation';

const GeneratedText = ({ text }) => (
  <div>
    {text && (
      <TypeAnimation
        sequence={[text]}
        wrapper="p"
        cursor={true}
        repeat={1}
        style={{ fontSize: '1em', whiteSpace: 'pre-wrap' }}
      />
    )}
  </div>
);

export default GeneratedText;