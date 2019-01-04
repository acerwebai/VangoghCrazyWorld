/*
===
Fast Style Transfer Simple Demo
===
*/

let nets = {};
let modelNames = ['Wheatfield_with_Crows', 'sien_with_a_cigar', 'Soup-Distribution', 'self-portrait', 'Red-Vineyards', 'la-campesinos', 'bedroom', 'Sunflowers-Bew','starrynight'];
let inputImg, styleImg;
let outputImgData;
let outputImg;
let modelNum = 0;
let currentModel = 'starrynight';
let uploader;
let uploader1;
let webcam = false;
let modelReady = false;
let video;
let start = false;
let isLoading = true;
let isSafa = false;

function setup() {
  isSafa = isSafari();
  if (isSafa) {
    alert('Sorry we do not yet support your device, please open this page with Chrome on a desktop. We will support other devices in the near future!');
    return;
  }

  noCanvas();
  inputImg = select('#input-img').elt;
//  inputImg1 = select('#input-img1').elt;
  styleImg = select('#style-img').elt;

  // load models
  modelNames.forEach(n => {
    nets[n] = new ml5.TransformNet('models/' + n + '/', modelLoaded);
  });

  // Image uploader
  uploader = select('#uploader').elt;
  uploader.addEventListener('change', gotNewInputImg);

//  uploader1 = select('#uploader1').elt;
//  uploader1.addEventListener('change', gotNewInputImg1);
    
  // output img container
  outputImgContainer = createImg('images/loading.gif', 'image');
  outputImgContainer.parent('output-img-container');

  allowFirefoxGetCamera();
}

// A function to be called when the model has been loaded
function modelLoaded() {
  modelNum++;
  if (modelNum >= modelNames.length) {
    modelReady = true;
    predictImg(currentModel);
  }
}

function predictImg(modelName) {
  isLoading = true;
  if (!modelReady) return;
  if (webcam && video) {
    outputImgData = nets[modelName].predict(video.elt);
  } else if (inputImg) {
    outputImgData = nets[modelName].predict(inputImg);
  }
//    else if(inputImg1) {
//      outputImgData = nets[modelName].predict(inputImg1);
//  }
  outputImg = ml5.array3DToImage(outputImgData);
  outputImgContainer.elt.src = outputImg.src;
  isLoading = false;
}

function draw() {
  if (modelReady && webcam && video && video.elt && start) {
    predictImg(currentModel);
  }
}

function updateStyleImg(ele) {
  if (ele.src) {
    styleImg.src = ele.src;
    currentModel = ele.id;
  }
  if (currentModel) {
    predictImg(currentModel);
  }
}

function updateInputImg(ele) {
  deactiveWebcam();
  if (ele.src) inputImg.src = ele.src;
  predictImg(currentModel);
}

function uploadImg() {
  uploader.click();
  deactiveWebcam();
}

function gotNewInputImg() {
  if (uploader.files && uploader.files[0]) {
    let newImgUrl = window.URL.createObjectURL(uploader.files[0]);
    inputImg.src = newImgUrl;
    inputImg.style.width = '200px';
    inputImg.style.height = '200px';
  }
}

function gotNewInputImg1() {
  if (uploader1.files && uploader1.files[0]) {
    let newImgUrl1 = window.URL.createObjectURL(uploader1.files[0]);
    inputImg1.src = newImgUrl1;
    inputImg1.style.width = '200px';
    inputImg1.style.height = '200px';
  }
}

function useWebcam() {
  if (!video) {
    // webcam video
    video = createCapture(VIDEO);
    video.size(180, 180);
    video.parent('input-source2');
  }
  webcam = true;
  select('#input-img2').hide();
  outputImgContainer.addClass('reverse-img');
}

function deactiveWebcam1() {
  start = false;
  select('#input-img1').show();
  outputImgContainer.removeClass('reverse-img');
  webcam = false;
  if (video) {
    video.hide();
    video = '';
  }
}

function deactiveWebcam() {
  start = false;
  select('#input-img2').show();
  outputImgContainer.removeClass('reverse-img');
  webcam = false;
  if (video) {
    video.hide();
    video = '';
  }
}

function onPredictClick() {
  if (webcam) start = true;
  predictImg(currentModel);
}

function allowFirefoxGetCamera() {
  navigator.getUserMedia = ( navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
}

function isSafari() {
  var ua = navigator.userAgent.toLowerCase();
  if (ua.indexOf('safari') != -1) {
    if (ua.indexOf('chrome') > -1) {
      return false;
    } else {
      return true;
    }
  }
}

/**
* @param imgData Array3D containing pixels of a img
* @return p5 Image
*/
// function array3DToP5Image(imgData) {  
//   const imgWidth = imgData.shape[0];
//   const imgHeight = imgData.shape[1];
//   const data = imgData.dataSync();
//   const outputImg = createImage(imgWidth, imgHeight);
//   outputImg.loadPixels();
//   let k = 0;
//   for (let i = 0; i < outputImg.width; i++) {
//     for (let j = 0; j < outputImg.height; j++) {
//       k = (i + j * height) * 3;
//       let r = floor(256 * data[k + 0]);
//       let g = floor(256 * data[k + 1]);
//       let b = floor(256 * data[k + 2]);
//       let c = color(r, g, b);
//       outputImg.set(i, j, c);
//     }
//   }
//   outputImg.updatePixels();
//   return outputImg;
// }
