import React, { useEffect, useState, useMemo } from "react";
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Tooltip,
} from "@mui/material";
import * as d3 from "d3";
import { analyzeSimilarity, getImageUrl } from "../api";

const SimilarityVisualizer = ({
  queryPath,
  resultPath,
  model,
  viewMode, // 'max' or 'pairwise'
}) => {
  const [loading, setLoading] = useState(false);
  const [matrix, setMatrix] = useState(null);
  const [gridSize, setGridSize] = useState(16);
  const [hoveredQueryIdx, setHoveredQueryIdx] = useState(null);
  const [hoveredResultIdx, setHoveredResultIdx] = useState(null);

  useEffect(() => {
    if (!queryPath || !resultPath) return;

    const fetchData = async () => {
      setLoading(true);
      try {
        const data = await analyzeSimilarity(queryPath, resultPath, model);
        if (data && data.matrix) {
          setMatrix(data.matrix);
          setGridSize(data.grid_size || 16);
        }
      } catch (error) {
        console.error("Analysis failed", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [queryPath, resultPath, model]);

  // Derived Heatmap Data
  const heatmapData = useMemo(() => {
    if (!matrix || !gridSize) return null;

    const size = gridSize * gridSize; // Total patches

    // Check if matrix is valid
    if (!Array.isArray(matrix) || matrix.length === 0) return null;

    let map = new Array(size).fill(0);

    try {
      if (viewMode === "max") {
        // Compute max for each result patch (axis 0)
        for (let r = 0; r < size; r++) {
          let maxVal = -1;
          for (let q = 0; q < size; q++) {
            // Safety check: ensure matrix[q] exists
            if (matrix[q] && matrix[q][r] !== undefined) {
              if (matrix[q][r] > maxVal) maxVal = matrix[q][r];
            }
          }
          map[r] = maxVal;
        }
      } else {
        // Pairwise mode
        if (hoveredQueryIdx !== null && matrix[hoveredQueryIdx]) {
          map = matrix[hoveredQueryIdx];
        } else {
          map = new Array(size).fill(0);
        }
      }
    } catch (e) {
      console.error("Error calculating heatmap", e);
      return null;
    }

    return map;
  }, [matrix, viewMode, hoveredQueryIdx, gridSize]);

  // Color Scale
  const colorScale = useMemo(() => {
    // Use d3.interpolateViridis which is standard in d3-scale-chromatic (often included or standard fallback)
    // interpolateJet is removed in newer d3 versions.
    try {
      return d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
    } catch (e) {
      console.error("D3 interpolation error", e);
      // Fallback simple scale
      return (val) => `rgba(255, 0, 0, ${val})`;
    }
  }, []);

  if (!queryPath || !resultPath) {
    return (
      <Typography variant="body1">
        Select a sample and perform a search to start.
      </Typography>
    );
  }

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        alignItems: "center",
        width: "100%",
      }}
    >
      {/* Query Image Area */}
      <Paper
        elevation={3}
        sx={{ p: 1, position: "relative", width: "fit-content" }}
      >
        <Typography variant="subtitle2" align="center">
          Query Image
        </Typography>
        <Box sx={{ position: "relative", width: 300, height: 300 }}>
          <img
            src={getImageUrl(queryPath)}
            alt="Query"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
              display: "block",
            }}
          />

          {/* Query Interaction Grid (Only relevant in Pairwise mode) */}
          {viewMode === "pairwise" && (
            <Box
              sx={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: "grid",
                gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
                gridTemplateRows: `repeat(${gridSize}, 1fr)`,
                cursor: "crosshair",
              }}
              onMouseLeave={() => setHoveredQueryIdx(null)}
            >
              {Array.from({ length: gridSize * gridSize }).map((_, idx) => (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredQueryIdx(idx)}
                  style={{
                    border:
                      hoveredQueryIdx === idx ? "1px solid white" : "none",
                    backgroundColor:
                      hoveredQueryIdx === idx
                        ? "rgba(255, 255, 255, 0.3)"
                        : "transparent",
                  }}
                />
              ))}
            </Box>
          )}
        </Box>
        {viewMode === "pairwise" && (
          <Typography
            variant="caption"
            display="block"
            align="center"
            color="text.secondary"
          >
            Hover patches here to see similarity on Result
          </Typography>
        )}
      </Paper>

      {/* Result Image Area */}
      <Paper
        elevation={3}
        sx={{ p: 1, position: "relative", width: "fit-content" }}
      >
        <Typography
          variant="subtitle2"
          align="center"
          sx={{ minHeight: "1.5rem" }}
        >
          Result Image
          {hoveredResultIdx !== null &&
            heatmapData &&
            heatmapData[hoveredResultIdx] !== undefined && (
              <span style={{ marginLeft: 10, fontWeight: "bold" }}>
                {model === "attention" ? "Attn" : "Sim"}:{" "}
                {heatmapData[hoveredResultIdx].toFixed(3)}
              </span>
            )}
        </Typography>

        <Box sx={{ position: "relative", width: 300, height: 300 }}>
          {loading ? (
            <Box
              sx={{
                display: "flex",
                height: "100%",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <CircularProgress />
            </Box>
          ) : (
            <>
              <img
                src={getImageUrl(resultPath)}
                alt="Result"
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "contain",
                  display: "block",
                }}
              />

              {/* Interactive Layer for Result Image */}
              <Box
                sx={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  display: "grid",
                  gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
                  gridTemplateRows: `repeat(${gridSize}, 1fr)`,
                  opacity: 0.6, // Apply opacity to the whole grid if it contains colors
                }}
                onMouseLeave={() => setHoveredResultIdx(null)}
              >
                {heatmapData &&
                  heatmapData.map((val, idx) => (
                    <div
                      key={idx}
                      onMouseEnter={() => setHoveredResultIdx(idx)}
                      style={{
                        backgroundColor: colorScale
                          ? colorScale(val)
                          : `rgba(255,0,0,${val})`,
                        border:
                          hoveredResultIdx === idx ? "1px solid white" : "none",
                      }}
                    />
                  ))}
              </Box>
            </>
          )}
        </Box>
      </Paper>

      {/* Legend */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, width: 300 }}>
        <Typography variant="caption">Low</Typography>
        <div
          style={{
            flex: 1,
            height: 10,
            background:
              "linear-gradient(to right, purple, blue, green, yellow)",
          }}
        />
        <Typography variant="caption">High</Typography>
      </Box>
    </Box>
  );
};

export default SimilarityVisualizer;
