import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemText,
} from "@mui/material";

const PredictionResult = ({ result }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h5" sx={{ color: "primary.main", mb: 2 }}>
          Prediction Results
        </Typography>
        <Typography variant="body1" sx={{ mb: 1 }}>
          <strong>Disease:</strong> {result.predicted_class}
        </Typography>
        <Typography variant="body1" sx={{ mb: 2 }}>
          <strong>Confidence:</strong> {result.confidence.toFixed(2)}%
        </Typography>
        {result.recommended_actions && result.recommended_actions.length > 0 ? (
          <>
            <Typography variant="h6" sx={{ mb: 1 }}>
              Recommended Actions:
            </Typography>
            <List>
              {result.recommended_actions.map((action, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemText
                    primary={`• ${action}`}
                    primaryTypographyProps={{ fontSize: "0.95rem" }}
                  />
                </ListItem>
              ))}
            </List>
          </>
        ) : (
          <Typography variant="body1" sx={{ mb: 2 }}>
            No recommended actions available.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default PredictionResult;
