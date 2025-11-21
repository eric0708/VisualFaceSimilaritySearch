import React from "react";
import { Box, Typography, Paper } from "@mui/material";
import { getImageUrl } from "../api";

const Gallery = ({ results, selectedResult, onSelectResult }) => {
  return (
    <Box sx={{ width: "100%", mt: 4 }}>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        Similar Results
      </Typography>
      <Box
        sx={{
          display: "flex",
          overflowX: "auto",
          gap: 2,
          pb: 2,
          "&::-webkit-scrollbar": { height: 6 },
          "&::-webkit-scrollbar-thumb": {
            backgroundColor: "rgba(74, 59, 50, 0.1)",
            borderRadius: 3,
          },
        }}
      >
        {results.map((result, index) => {
          const isSelected =
            selectedResult && result.path === selectedResult.path;
          return (
            <Paper
              key={index}
              elevation={0}
              sx={{
                minWidth: 140,
                maxWidth: 140,
                cursor: "pointer",
                border: isSelected
                  ? "2px solid #D97757"
                  : "1px solid rgba(74, 59, 50, 0.16)",
                borderRadius: 2,
                transition: "all 0.2s",
                overflow: "hidden",
                bgcolor: "background.paper",
                "&:hover": {
                  transform: "translateY(-2px)",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
                },
              }}
              onClick={() => onSelectResult(result)}
            >
              <Box sx={{ width: "100%", height: 140, overflow: "hidden" }}>
                <img
                  src={getImageUrl(result.path)}
                  alt={`Result ${index}`}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
              </Box>
              <Box sx={{ p: 1.5 }}>
                <Typography
                  variant="caption"
                  display="block"
                  align="center"
                  sx={{ fontWeight: 600, color: "text.secondary" }}
                >
                  {result.score % 1 === 0 && result.score <= 12
                    ? `Layer ${Math.round(result.score)}`
                    : `Sim: ${result.score.toFixed(3)}`}
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
