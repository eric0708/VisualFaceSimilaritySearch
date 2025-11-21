import React, { useState, useEffect } from "react";
import {
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Alert,
  Snackbar,
  CircularProgress,
  Typography,
} from "@mui/material";
import Sidebar from "./components/Sidebar";
import SimilarityVisualizer from "./components/SimilarityVisualizer";
import Gallery from "./components/Gallery";
import { fetchSamples, fetchModels, searchImage, loadModel } from "./api";

const theme = createTheme({
  typography: {
    fontFamily: [
      "Inter",
      "-apple-system",
      "BlinkMacSystemFont",
      '"Segoe UI"',
      "Roboto",
      '"Helvetica Neue"',
      "Arial",
      "sans-serif",
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(","),
    h4: {
      fontWeight: 700,
      color: "#4A3B32",
      letterSpacing: "-0.03em",
    },
    h6: {
      fontWeight: 600,
      color: "#4A3B32",
      letterSpacing: "-0.01em",
    },
    subtitle2: {
      fontWeight: 600,
      color: "rgba(74, 59, 50, 0.7)",
      fontSize: "0.75rem",
      textTransform: "uppercase",
      letterSpacing: "0.04em",
    },
    body1: {
      color: "#4A3B32",
    },
  },
  palette: {
    mode: "light",
    primary: {
      main: "#D97757", // Warm Terra Cotta
    },
    text: {
      primary: "#4A3B32", // Dark Warm Brown
      secondary: "rgba(74, 59, 50, 0.7)", // Muted Warm Brown
    },
    background: {
      default: "#FEFDF5", // Soft Cream
      paper: "#FFFFFF",
    },
    divider: "rgba(74, 59, 50, 0.1)",
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: "#FEFDF5",
          color: "#4A3B32",
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 600,
          borderRadius: 8,
          boxShadow: "none",
          "&:hover": {
            boxShadow: "none",
            backgroundColor: "rgba(217, 119, 87, 0.08)",
          },
        },
        contained: {
          backgroundColor: "#D97757",
          color: "#FFFFFF",
          "&:hover": {
            backgroundColor: "#B95E40",
            boxShadow: "0 4px 12px rgba(217, 119, 87, 0.2)",
          },
        },
        outlined: {
          borderColor: "rgba(217, 119, 87, 0.5)",
          color: "#D97757",
          "&:hover": {
            borderColor: "#D97757",
            backgroundColor: "rgba(217, 119, 87, 0.04)",
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: "0 2px 12px rgba(74, 59, 50, 0.03)",
          border: "1px solid rgba(74, 59, 50, 0.05)",
        },
        elevation0: {
          boxShadow: "none",
          border: "none",
        },
        elevation3: {
          boxShadow: "0 8px 24px rgba(74, 59, 50, 0.08)",
          border: "none",
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderColor: "rgba(74, 59, 50, 0.2)",
          "&:hover": {
            borderColor: "rgba(74, 59, 50, 0.5)",
          },
        },
      },
    },
  },
});

function App() {
  const [samples, setSamples] = useState([]);
  const [models, setModels] = useState([]);

  // Loading States
  const [selectedModel, setSelectedModel] = useState(""); // Start empty
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isSearching, setIsSearching] = useState(false);

  const [viewMode, setViewMode] = useState("max");

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
          fetchModels(),
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
      setSelectedModel(""); // Reset
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
      formData.append("model", selectedModel);

      if (type === "file") {
        formData.append("image", input);
      } else {
        formData.append("image_path", input);
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
      <Box sx={{ display: "flex", height: "100vh", overflow: "hidden" }}>
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
          onSelectSample={(path) => handleSearch(path, "path")}
          onUpload={(file) => handleSearch(file, "file")}
        />

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          {!selectedModel ? (
            <Box
              sx={{
                m: "auto",
                textAlign: "center",
                color: "text.secondary",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Typography variant="h4">Welcome</Typography>
              <Typography variant="body1" sx={{ mt: 2 }}>
                Please select a model from the sidebar to begin.
              </Typography>
            </Box>
          ) : isModelLoading ? (
            <Box
              sx={{
                m: "auto",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <CircularProgress size={60} />
              <Typography variant="h6" sx={{ mt: 2 }}>
                Loading Model Resources...
              </Typography>
            </Box>
          ) : isSearching ? (
            <Box
              sx={{
                m: "auto",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <CircularProgress size={60} />
              <Typography variant="h6" sx={{ mt: 2 }}>
                Searching for similar faces...
              </Typography>
            </Box>
          ) : queryPath && selectedResult ? (
            <Box
              sx={{
                m: "auto",
                width: "100%",
                maxWidth: 1000,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
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
            </Box>
          ) : (
            <Box
              sx={{
                m: "auto",
                textAlign: "center",
                color: "text.secondary",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Typography variant="h5">
                Select a sample image or upload one to search
              </Typography>
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
