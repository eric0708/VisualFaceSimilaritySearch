import React, { useState, useEffect } from 'react';
import { Box, CssBaseline, ThemeProvider, createTheme, Alert, Snackbar, CircularProgress, Typography } from '@mui/material';
import Sidebar from './components/Sidebar';
import SimilarityVisualizer from './components/SimilarityVisualizer';
import Gallery from './components/Gallery';
import { fetchSamples, fetchModels, searchImage, loadModel } from './api';

const theme = createTheme({
  palette: {
    primary: { main: '#1976d2' },
    secondary: { main: '#dc004e' },
  },
});

function App() {
  const [samples, setSamples] = useState([]);
  const [models, setModels] = useState([]);
  
  // Loading States
  const [selectedModel, setSelectedModel] = useState(''); // Start empty
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  
  const [viewMode, setViewMode] = useState('max');
  
  const [queryPath, setQueryPath] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  
  const [error, setError] = useState(null);

  useEffect(() => {
    // Initial load
    const init = async () => {
      try {
        const [samplesData, modelsData] = await Promise.all([
          fetchSamples(),
          fetchModels()
        ]);
        setSamples(samplesData || []);
        if (modelsData && modelsData.length > 0) {
          setModels(modelsData);
        }
      } catch (err) {
        console.error("Initialization error", err);
        setError("Failed to load initial data. Is backend running?");
      }
    };
    init();
  }, []);

  const handleModelSelect = async (modelId) => {
    if (modelId === selectedModel && isModelLoaded) return;
    
    setSelectedModel(modelId);
    setIsModelLoading(true);
    setIsModelLoaded(false);
    
    setSearchResults([]);
    setSelectedResult(null);
    setQueryPath(null);
    
    try {
      await loadModel(modelId);
      setIsModelLoaded(true);
    } catch (err) {
      console.error("Model load failed", err);
      setError(`Failed to load model: ${modelId}`);
      setSelectedModel(''); // Reset
    } finally {
      setIsModelLoading(false);
    }
  };

  const handleSearch = async (input, type) => {
    setSearchResults([]);
    setSelectedResult(null);
    setQueryPath(null);
    setIsSearching(true);

    try {
      const formData = new FormData();
      formData.append('model', selectedModel);
      
      if (type === 'file') {
        formData.append('image', input);
      } else {
        formData.append('image_path', input);
      }
      
      const data = await searchImage(formData);
      setQueryPath(data.query_path);
      setSearchResults(data.results);
      
      // Select first result by default
      if (data.results && data.results.length > 0) {
        setSelectedResult(data.results[0]);
      } else {
        setSelectedResult(null);
      }
    } catch (err) {
      console.error("Search error", err);
      setError("Search failed. Please try again.");
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
        
        {/* Sidebar */}
        <Sidebar
          samples={samples}
          models={models}
          selectedModel={selectedModel}
          onSelectModel={handleModelSelect}
          isModelLoading={isModelLoading}
          isModelLoaded={isModelLoaded}
          
          viewMode={viewMode}
          onToggleMode={setViewMode}
          onSelectSample={(path) => handleSearch(path, 'path')}
          onUpload={(file) => handleSearch(file, 'file')}
        />

        {/* Main Content */}
        <Box 
          component="main" 
          sx={{ 
            flexGrow: 1, 
            p: 3, 
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center'
          }}
        >
          {!selectedModel ? (
            <Box sx={{ mt: 10, textAlign: 'center', color: 'text.secondary' }}>
              <Typography variant="h4">Welcome</Typography>
              <Typography variant="body1" sx={{ mt: 2 }}>
                Please select a model from the sidebar to begin.
              </Typography>
            </Box>
          ) : isModelLoading ? (
            <Box sx={{ mt: 20, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <CircularProgress size={60} />
              <Typography variant="h6" sx={{ mt: 2 }}>Loading Model Resources...</Typography>
            </Box>
          ) : isSearching ? (
            <Box sx={{ mt: 20, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <CircularProgress size={60} />
              <Typography variant="h6" sx={{ mt: 2 }}>Searching for similar faces...</Typography>
            </Box>
          ) : queryPath && selectedResult ? (
            <>
              <SimilarityVisualizer
                queryPath={queryPath}
                resultPath={selectedResult.path}
                model={selectedModel}
                viewMode={viewMode}
              />
              
              <Gallery
                results={searchResults}
                selectedResult={selectedResult}
                onSelectResult={setSelectedResult}
              />
            </>
          ) : (
            <Box sx={{ mt: 10, textAlign: 'center', color: 'text.secondary' }}>
              <h2>Select a sample image or upload one to search</h2>
            </Box>
          )}
        </Box>
      </Box>

      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError(null)}
      >
        <Alert onClose={() => setError(null)} severity="error">
          {error}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App;
