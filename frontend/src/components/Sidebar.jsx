import React from "react";
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
  CircularProgress,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { getImageUrl } from "../api";

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
  isModelLoaded,
}) => {
  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      onUpload(event.target.files[0]);
    }
  };

  return (
    <Paper
      elevation={0}
      square
      sx={{
        width: 300,
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        p: 3,
        overflowY: "auto",
        backgroundColor: "#F2EBE3", // Warm Beige
        borderRight: "1px solid rgba(74, 59, 50, 0.08)",
      }}
    >
      <Typography
        variant="h6"
        gutterBottom
        sx={{ mb: 3, fontWeight: 700, color: "text.primary" }}
      >
        Visual Search
      </Typography>

      {/* Model Selection */}
      <FormControl fullWidth sx={{ mb: 3 }} size="small">
        <InputLabel
          id="model-select-label"
          sx={{ bgcolor: "#F2EBE3", px: 0.5, color: "text.secondary" }}
        >
          Methods
        </InputLabel>
        <Select
          labelId="model-select-label"
          value={selectedModel || ""}
          onChange={(e) => onSelectModel(e.target.value)}
          disabled={isModelLoading}
          variant="outlined"
        >
          {models.map((m) => (
            <MenuItem key={m.id} value={m.id}>
              {m.name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {isModelLoading && (
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            mb: 3,
          }}
        >
          <CircularProgress size={24} sx={{ mb: 1 }} />
          <Typography variant="caption">Loading Model...</Typography>
        </Box>
      )}

      {isModelLoaded && (
        <>
          {/* View Mode Toggle - Hide for Grad-CAM and Attention */}
          {selectedModel !== "gradcam" && selectedModel !== "attention" && (
            <>
              <Typography
                variant="subtitle2"
                gutterBottom
                sx={{ mt: 2, mb: 1 }}
              >
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
            </>
          )}

          <Divider sx={{ mb: 3, opacity: 0.6 }} />

          {/* Upload */}
          <Button
            component="label"
            variant="outlined" // Changed to outlined for cleaner look
            startIcon={<CloudUploadIcon />}
            fullWidth
            sx={{
              mb: 3,
              bgcolor: "background.paper",
              borderColor: "rgba(74, 59, 50, 0.2)",
            }}
          >
            Upload Image
            <input
              type="file"
              hidden
              accept="image/*"
              onChange={handleFileChange}
            />
          </Button>

          {/* Sample Images */}
          <Typography variant="subtitle2" gutterBottom sx={{ mb: 1 }}>
            Sample Images
          </Typography>
          <ImageList cols={3} rowHeight={80} gap={8}>
            {samples.map((path, index) => (
              <ImageListItem
                key={index}
                sx={{
                  cursor: "pointer",
                  borderRadius: 2,
                  overflow: "hidden",
                  border: "1px solid transparent",
                  transition: "all 0.2s",
                  "&:hover": {
                    opacity: 0.8,
                    transform: "translateY(-2px)",
                    borderColor: "primary.main",
                  },
                }}
                onClick={() => onSelectSample(path)}
              >
                <img
                  src={`${getImageUrl(path)}`}
                  alt={`Sample ${index}`}
                  loading="lazy"
                  style={{ height: "100%", objectFit: "cover" }}
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
