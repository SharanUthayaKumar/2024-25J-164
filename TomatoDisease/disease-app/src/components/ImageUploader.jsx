import {
  Button,
  Box,
  Paper,
  Typography,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';

const ImageUploader = ({ onImageChange, imagePreview, onPredict }) => {
  return (
    <Paper
      elevation={3}
      sx={{
        p: 3,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        background: 'linear-gradient(135deg, #ffffff, #e3f2fd)',
      }}
    >
      <Typography variant="h6" sx={{ mb: 2 }}>
        Upload Tomato Leaf Image
      </Typography>
      <Button
        variant="contained"
        component="label"
        startIcon={<UploadFileIcon />}
        sx={{ mb: 2 }}
      >
        Choose Image
        <input
          type="file"
          hidden
          accept="image/*"
          onChange={onImageChange}
        />
      </Button>
      {imagePreview && (
        <Box
          component="img"
          src={imagePreview}
          alt="Selected"
          sx={{
            maxWidth: '100%',
            maxHeight: 300,
            borderRadius: 2,
            mt: 2,
            boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          }}
        />
      )}
      <Button
        variant="contained"
        onClick={onPredict}
        disabled={!imagePreview}
        sx={{ mt: 2, minWidth: 120 }}
      >
        Predict
      </Button>
    </Paper>
  );
};

export default ImageUploader;