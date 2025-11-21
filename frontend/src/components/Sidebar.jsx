import React from 'react';
import { 
  Box, 
  Typography, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel, 
  ToggleButton, 
  ToggleButtonGroup,
  Button,
  ImageList,
  ImageListItem,
  Paper,
  Divider,
  CircularProgress
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { getImageUrl } from '../api';

const Sidebar = ({ 
  samples, 
  onSelectSample, 
  onUpload, 
  models, 
  selectedModel, 
  onSelectModel, 
  viewMode, 
  onToggleMode,
  isModelLoading,
  isModelLoaded 
}) => {
  
  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      onUpload(event.target.files[0]);
    }
  };

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        width: 300, 
        height: '100vh', 
        display: 'flex', 
        flexDirection: 'column', 
        p: 2, 
        overflowY: 'auto',
        backgroundColor: '#f5f5f5'
      }}
    >
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: '#1976d2' }}>
        Settings
      </Typography>

      {/* Model Selection */}
      <FormControl fullWidth sx={{ mb: 3 }} size="small">
        <InputLabel>Model</InputLabel>
        <Select
          value={selectedModel || ''}
          label="Model"
          onChange={(e) => onSelectModel(e.target.value)}
          disabled={isModelLoading}
        >
          {models.map(m => (
            <MenuItem key={m.id} value={m.id}>{m.name}</MenuItem>
          ))}
        </Select>
      </FormControl>

      {isModelLoading && (
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 3 }}>
          <CircularProgress size={24} sx={{ mb: 1 }} />
          <Typography variant="caption">Loading Model...</Typography>
        </Box>
      )}

      {isModelLoaded && (
        <>
          {/* View Mode Toggle */}
          <Typography variant="subtitle2" gutterBottom>
            Attention Mode
          </Typography>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(e, newMode) => newMode && onToggleMode(newMode)}
            fullWidth
            size="small"
            sx={{ mb: 3 }}
          >
            <ToggleButton value="max">Max</ToggleButton>
            <ToggleButton value="pairwise">Pairwise</ToggleButton>
          </ToggleButtonGroup>

          <Divider sx={{ mb: 2 }} />

          {/* Upload */}
          <Button
            component="label"
            variant="contained"
            startIcon={<CloudUploadIcon />}
            fullWidth
            sx={{ mb: 3 }}
          >
            Upload Image
            <input type="file" hidden accept="image/*" onChange={handleFileChange} />
          </Button>

          {/* Sample Images */}
          <Typography variant="subtitle2" gutterBottom>
            Sample Images
          </Typography>
          <ImageList cols={3} rowHeight={80} gap={8}>
            {samples.map((path, index) => (
              <ImageListItem 
                key={index} 
                sx={{ 
                  cursor: 'pointer',
                  border: '1px solid #ddd',
                  '&:hover': { opacity: 0.8, borderColor: '#1976d2' }
                }}
                onClick={() => onSelectSample(path)}
              >
                <img
                  src={`${getImageUrl(path)}`}
                  alt={`Sample ${index}`}
                  loading="lazy"
                  style={{ height: '100%', objectFit: 'cover' }}
                />
              </ImageListItem>
            ))}
          </ImageList>
        </>
      )}
    </Paper>
  );
};

export default Sidebar;
