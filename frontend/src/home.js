import { useState, useEffect } from "react";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Typography from "@material-ui/core/Typography";
import Container from "@material-ui/core/Container";
import React from "react";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { Paper, CardActionArea, CardMedia, Grid, TableContainer, Table, TableBody, TableHead, TableRow, TableCell, Button, CircularProgress } from "@material-ui/core";
import { DropzoneArea } from 'material-ui-dropzone';
import { common } from '@material-ui/core/colors';
import Clear from '@material-ui/icons/Clear';
import axios from 'axios';

const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText(common.white),
    backgroundColor: common.white,
    '&:hover': {
      backgroundColor: '#ffffff7a',
    },
  },
}))(Button);

const useStyles = makeStyles((theme) => ({
  grow: { flexGrow: 1 },
  clearButton: {
    width: "100%",
    borderRadius: "15px",
    padding: "15px 22px",
    color: "#000000a6",
    fontSize: "20px",
    fontWeight: 900,
  },
  root: { maxWidth: 345, flexGrow: 1 },
  media: { height: 400 },
  paper: { padding: theme.spacing(2), margin: 'auto', maxWidth: 500 },
  gridContainer: { justifyContent: "center", padding: "4em 1em 0 1em" },
  mainContainer: {
    backgroundColor: 'black',
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    height: "93vh",
    marginTop: "8px",
  },
  imageCard: {
    margin: "auto",
    maxWidth: 400,
    height: 500,
    backgroundColor: 'transparent',
    boxShadow: '0px 9px 70px 0px rgb(0 0 0 / 30%) !important',
    borderRadius: '15px',
  },
  imageCardEmpty: { height: 'auto' },
  noImage: { margin: "auto", width: 400, height: "400 !important" },
  input: { display: 'none' },
  tableContainer: { backgroundColor: 'transparent', boxShadow: 'none' },
  tableCell: {
    fontSize: '22px',
    backgroundColor: 'transparent',
    color: '#000000a6',
    fontWeight: 'bolder',
    padding: '1px 24px 1px 16px',
  },
  tableCell1: {
    fontSize: '14px',
    backgroundColor: 'transparent',
    color: '#000000a6',
    fontWeight: 'bolder',
    padding: '1px 24px 1px 16px',
  },
  tableBody: { backgroundColor: 'transparent' },
  text: { color: 'white !important', textAlign: 'center' },
  appbar: { background: '#be6a77', boxShadow: 'none', color: 'white' },
  loader: { color: '#be6a77 !important' },
}));

export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [image, setImage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  let confidence = 0;

  const sendFile = async () => {
    if (image) {
      try {
        let formData = new FormData();
        formData.append("file", selectedFile);

        let res = await axios.post(`https://classifier-leaves.onrender.com/predict`, formData);

        if (res.status === 200) {
          setData(res.data);
        }
      } catch (error) {
        console.error("Error uploading file: ", error);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const clearData = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    
    // Cleanup URL on unmount to avoid memory leaks
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) return;

    setIsLoading(true);
    sendFile();
  }, [preview]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(undefined);
      setImage(false);
      setData(undefined);
      return;
    }
    setSelectedFile(files[0]);
    setData(undefined);
    setImage(true);
  };

  if (data) {
    confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  }

  return (
    <React.Fragment>
      <AppBar position="static" className={classes.appbar} />
      <Container maxWidth={false} className={classes.mainContainer} disableGutters>
        <Grid container className={classes.gridContainer} justifyContent="center" alignItems="center" spacing={2}>
          <Grid item xs={12}>
            <Card className={`${classes.imageCard} ${!image ? classes.imageCardEmpty : ''}`}>
              {image && (
                <CardActionArea>
                  <CardMedia className={classes.media} image={preview} component="img" title="Selected Image" />
                </CardActionArea>
              )}
              {!image && (
                <CardContent className={classes.content}>
                  <DropzoneArea acceptedFiles={['image/*']} dropzoneText={"Drag and drop an image here"} onChange={onSelectFile} />
                </CardContent>
              )}
              {data && (
                <CardContent className={classes.detail}>
                  <TableContainer component={Paper} className={classes.tableContainer}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell className={classes.tableCell1}>Label:</TableCell>
                          <TableCell align="right" className={classes.tableCell1}>Confidence:</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell className={classes.tableCell}>{data.class}</TableCell>
                          <TableCell align="right" className={classes.tableCell}>{confidence}%</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              )}
              {isLoading && (
                <CardContent className={classes.detail}>
                  <CircularProgress color="secondary" className={classes.loader} />
                  <Typography variant="h6">Processing</Typography>
                </CardContent>
              )}
            </Card>
          </Grid>
          {data && (
            <Grid item className={classes.buttonGrid}>
              <ColorButton variant="contained" className={classes.clearButton} onClick={clearData} startIcon={<Clear fontSize="large" />}>
                Clear
              </ColorButton>
            </Grid>
          )}
        </Grid>
      </Container>
    </React.Fragment>
  );
};
