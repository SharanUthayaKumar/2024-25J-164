import { useState } from "react";
import { Container, Typography, CircularProgress, Box } from "@mui/material";
import ImageUploader from "../components/ImageUploader";
import PredictionResult from "../components/PredictionResult";
import ErrorAlert from "../components/ErrorAlert";
import { predictDisease } from "../services/api";

const Home = () => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!image) {
      setError("Please select an image.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictDisease(image);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography
        variant="h3"
        align="center"
        gutterBottom
        sx={{ fontWeight: "bold", color: "primary.main", mb: 4 }}
      >
        Tomato Disease Predictor
      </Typography>

      <Box sx={{ mb: 4 }}>
        <ImageUploader
          onImageChange={handleImageChange}
          imagePreview={imagePreview}
          onPredict={handlePredict}
        />
      </Box>

      {loading && (
        <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Box sx={{ my: 4 }}>
          <ErrorAlert message={error} />
        </Box>
      )}

      {result && (
        <Box sx={{ my: 4 }}>
          <PredictionResult result={result} />
        </Box>
      )}
    </Container>
  );
};

export default Home;
