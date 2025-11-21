import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { getImageUrl } from '../api';

const Gallery = ({ results, selectedResult, onSelectResult }) => {
  return (
    <Box sx={{ width: '100%', mt: 2 }}>
      <Typography variant="h6" gutterBottom>
        Similar Results
      </Typography>
      <Box 
        sx={{ 
          display: 'flex', 
          overflowX: 'auto', 
          gap: 2, 
          pb: 2,
          '&::-webkit-scrollbar': { height: 8 },
          '&::-webkit-scrollbar-thumb': { backgroundColor: '#ccc', borderRadius: 4 }
        }}
      >
        {results.map((result, index) => {
           const isSelected = selectedResult && result.path === selectedResult.path;
           return (
            <Paper
              key={index}
              elevation={isSelected ? 6 : 2}
              sx={{
                minWidth: 120,
                maxWidth: 120,
                cursor: 'pointer',
                border: isSelected ? '3px solid #1976d2' : '1px solid transparent',
                transition: 'all 0.2s',
                overflow: 'hidden'
              }}
              onClick={() => onSelectResult(result)}
            >
              <Box sx={{ width: '100%', height: 120, overflow: 'hidden' }}>
                <img 
                  src={getImageUrl(result.path)} 
                  alt={`Result ${index}`}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }} 
                />
              </Box>
              <Box sx={{ p: 1, bgcolor: '#fff' }}>
                <Typography variant="caption" display="block" align="center" fontWeight="bold">
                  Sim: {result.score.toFixed(3)}
                </Typography>
              </Box>
            </Paper>
          );
        })}
      </Box>
    </Box>
  );
};

export default Gallery;

