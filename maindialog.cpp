#include "maindialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>


#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv\cxcore.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/video/background_segm.hpp>
#include <QClipboard>
#include <QApplication>
#include <QMimeData>

#define IMAGEWIDTH 400
#define IMAGEHEIGHT 300

#define PUZZLEPIECEWIDTH 50
#define PUZZLEPIECEHEIGHT 50

#define IMAGE_THRESHOLD 225
#define MASK_THRESHOLD 240
#define AREA_THRESHOLD 100

#define DEBUG 1


using namespace std;
using namespace cv;

namespace{

	//Ref:http://asmaloney.com/2013/11/code/converting-between-cvmat-and-qimage-or-qpixmap/
	inline cv::Mat QImageToCvMat( const QImage &inImage, bool inCloneImageData = true )
   {
      switch ( inImage.format() )
      {
         // 8-bit, 4 channel
         case QImage::Format_RGB32:
         {
            cv::Mat  mat( inImage.height(), inImage.width(), CV_8UC4, const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine() );
 
            return (inCloneImageData ? mat.clone() : mat);
         }
 
         // 8-bit, 3 channel
         case QImage::Format_RGB888:
         {
            if ( !inCloneImageData )
				std::cout << "ASM::QImageToCvMat() - Conversion requires cloning since we use a temporary QImage";
 
            QImage   swapped = inImage.rgbSwapped();
 
            return cv::Mat( swapped.height(), swapped.width(), CV_8UC3, const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine() ).clone();
         }
 
         // 8-bit, 1 channel
         case QImage::Format_Indexed8:
         {
            cv::Mat  mat( inImage.height(), inImage.width(), CV_8UC1, const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine() );
 
            return (inCloneImageData ? mat.clone() : mat);
         }
 
         default:
            std::cout << "ASM::QImageToCvMat() - QImage format not handled in switch:" << inImage.format();
            break;
      }
 
      return cv::Mat();
   }
inline cv::Mat QPixmapToCvMat( const QPixmap &inPixmap, bool inCloneImageData = true )
{
    return QImageToCvMat( inPixmap.toImage(), inCloneImageData );
}

cv::Mat  extractImageFromBackground(cv::Mat inputImg)
{
	int inputWidth = inputImg.rows;
	int inputHeight = inputImg.cols;	

	cv::Mat grayImg;
	cv::cvtColor(inputImg,grayImg, CV_BGR2GRAY);
	cv::threshold(grayImg,grayImg,IMAGE_THRESHOLD,255,0);

#if DEBUG
	imshow("gray scale",grayImg);
	resizeWindow("gray scale",200,200);
	moveWindow("gray scale", 100, 100);
#endif
	cv::Mat whiteBackground = cv::Mat(inputWidth,inputHeight,inputImg.type());
	whiteBackground.setTo(cv::Scalar(255,255,255));

#if DEBUG
	imshow("input image",inputImg);
	resizeWindow("input image",200,200);
	moveWindow("input image", 400, 100);
	//imshow("background",whiteBackground);
#endif
	

	Mat fgMaskMOG2 , output;
	Ptr<BackgroundSubtractor> pMOG2;
	pMOG2 = new BackgroundSubtractorMOG2();
	pMOG2->operator()(whiteBackground, fgMaskMOG2);
	pMOG2->operator()(grayImg, fgMaskMOG2);
	cv::threshold(fgMaskMOG2,fgMaskMOG2,MASK_THRESHOLD,255,0);
#if DEBUG
	imshow("mask",fgMaskMOG2);
	resizeWindow("mask",200,200);
	moveWindow("mask", 100, 400);
#endif
	std::cout << fgMaskMOG2.rows << fgMaskMOG2.cols;

	inputImg.copyTo(output, fgMaskMOG2);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	findContours(fgMaskMOG2,contours,hierarchy,cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS);

	cv::Rect brect ;
	for ( size_t i=0; i<contours.size(); ++i )
	{		
		brect = cv::boundingRect(contours[i]);
		if(brect.area() > AREA_THRESHOLD)
		{
			cv::drawContours( output, contours, i, Scalar(0,255,0), 1, 8, hierarchy, 0, Point() ); 
			break;		
		}
	}
#if DEBUG
	imshow("bounds",output);
	resizeWindow("bounds",200,200);
	moveWindow("bounds", 400, 400);
#endif
	cv::Mat finalOutput = cv::Mat(inputImg, cv::Range(brect.y,brect.y+brect.height),cv::Range(brect.x,brect.x + brect.width));	
#if DEBUG
	imshow("Extracted Image",finalOutput);
	resizeWindow("Extracted Image",200,200);
	moveWindow("Extracted Image",400, 650);
#endif
	return finalOutput;

}

}

MainDialog::MainDialog(QWidget *parent)
    : QDialog(parent)
{
    this->setFixedWidth(600);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    QGridLayout* gridLayout = new QGridLayout();

    mainLayout->addLayout(gridLayout);
    QLabel* uploadImageLabel = new QLabel("Upload Puzzle Image");
    gridLayout->addWidget(uploadImageLabel,0,0);
    QPushButton* uploadImageButton = new QPushButton("Choose..");
    gridLayout->addWidget(uploadImageButton,0,1);

    uploadImageButton->setSizePolicy(QSizePolicy::Fixed,QSizePolicy::Fixed);
    connect(uploadImageButton,SIGNAL(clicked()),this,SLOT(uploadImageClicked()));
    m_mainImage = new QLabel;
    m_mainImage->setMaximumSize(QSize(IMAGEWIDTH,IMAGEHEIGHT));
    gridLayout->addWidget(m_mainImage,0,2);
    mainLayout->addStretch();

    QLabel* selectPuzzlePiece = new QLabel("Select a Puzzle Piece");
    gridLayout->addWidget(selectPuzzlePiece,1,0);
    m_selectPieceButton = new QPushButton("Choose..");
    m_selectPieceButton->setEnabled(false);
    connect(m_selectPieceButton,SIGNAL(clicked()),this,SLOT(uploadPuzzlePieceClicked()));
    gridLayout->addWidget(m_selectPieceButton,1,1);
    m_puzzlePiece = new QLabel;
    m_puzzlePiece->setMaximumSize(QSize(IMAGEWIDTH,IMAGEHEIGHT));
    gridLayout->addWidget(m_puzzlePiece,1,2);
    mainLayout->addStretch();

    m_detectButton = new QPushButton("Detect Position");
	connect(m_detectButton,SIGNAL(clicked()),this,SLOT(process()));
    m_detectButton->setEnabled(false);
    gridLayout->addWidget(m_detectButton,2,0,1,3,Qt::AlignHCenter);

    m_result = new QLabel("");
	gridLayout->addWidget(m_result,3,0,1,3,Qt::AlignCenter);

    /*
    QVBoxLayout* mainLeftLayout = new QVBoxLayout(this);
    QVBoxLayout* mainRightLayout = new QVBoxLayout(this);


    mainLayout->addLayout(mainRightLayout);
    mainLeftLayout->addLayout(uploadImageLayout);
    mainLeftLayout->addLayout(uploadPieceLayout);


*/
}

MainDialog::~MainDialog()
{

}

void MainDialog::uploadPuzzlePieceClicked()
{
	m_puzzlePiecePath = QFileDialog::getOpenFileName(this,tr("Select Puzzle Imaage"),"",tr("Images (*.png *.jpg)"));
    if (!m_puzzlePiecePath.isEmpty())
    {
         QImage image(m_puzzlePiecePath);
         if (image.isNull()) {
             QMessageBox::information(this, tr("Image Viewer"),
                                      tr("Cannot load %1.").arg(m_puzzlePiecePath));
             return;
         }
         QPixmap pixmap1 = QPixmap::fromImage(image);
         m_puzzlePiece->setPixmap(pixmap1.scaled(QSize(PUZZLEPIECEWIDTH, PUZZLEPIECEHEIGHT)));
         m_detectButton->setEnabled(true);
		 
		cv::Mat inputPiece = cv::imread(m_puzzlePiecePath.toStdString());

		m_inputPiece = extractImageFromBackground(inputPiece);

     }
}

void MainDialog::uploadImageClicked()
{

    m_wholeImagePath = QFileDialog::getOpenFileName(this,tr("Select Puzzle Imaage"),"",tr("Images (*.png *.jpg)"));
    if (!m_wholeImagePath.isEmpty())
    {
         QImage image(m_wholeImagePath);
         if (image.isNull()) {
             QMessageBox::information(this, tr("Image Viewer"),
                                      tr("Cannot load %1.").arg(m_wholeImagePath));
             return;
         }
         QPixmap pixmap1 = QPixmap::fromImage(image);
         m_mainImage->setPixmap(pixmap1.scaled(QSize(IMAGEWIDTH, IMAGEHEIGHT)));
         m_selectPieceButton->setEnabled(true);
         //m_imageLabel->adjustSize();
         //m_imageLabel->resize(0.5 * m_imageLabel->pixmap()->size());
		 m_inputImage = cvLoadImage(m_wholeImagePath.toLocal8Bit(),1);   // whole image	
	
     }

}


void MainDialog::process()
{

    //Define images to store each frame and results after match with the templates
    IplImage* temp1MatchResult;
	Mat imageCopy = m_inputImage.clone();
    // Templates
	IplImage* frame =  new IplImage(imageCopy);
	IplImage* temp1 = new IplImage(m_inputPiece); // piece image

	int rowNum = frame->height/ temp1->height; // number of pieces in each row
	int colNum = frame->width/temp1->width; // number of pieces in each col

    // Initialize the images use to store the results after match with the templates
    int w1 = frame->width  - temp1->width  + 1;
    int h1 = frame->height - temp1->height + 1;
    int w2 = frame->width   + 1;
    int h2 = frame->height  + 1;

    cout<<w2<<"  "<<h2<<endl;


    int rowUnitHeight =0;
    int colUnitWidth = 0;
    if (rowNum != 0 && colNum != 0){
        rowUnitHeight = h2/rowNum;
        colUnitWidth = w2/colNum;
    }

    cout<<colUnitWidth<<" "<<rowUnitHeight<<endl;

    temp1MatchResult = cvCreateImage(cvSize(w1, h1), IPL_DEPTH_32F, 1);

    cvZero(temp1MatchResult);


    //Define a font to display number of hits
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1, 1.2, 0, 2);

    //Define window to display video 
    cvNamedWindow("Jigsaw puzzles solver", 0);

//    assert(frame) ;

    cvMatchTemplate(frame, temp1, temp1MatchResult, 5);


    if(temp1MatchResult){
        //Find a corner of the matched template to get the coordinates to draw the rectangles 
        double min_val1=0, max_val1=0, min_val2=0, max_val2=0;
        CvPoint min_loc1, max_loc1;

        cvMinMaxLoc(temp1MatchResult, &min_val1, &max_val1, &min_loc1, &max_loc1);


        int offsetx = 0;
        int offsety = 0;
        //Draw red color rectangle around the bird //cvR
        cvRectangle(frame,cvPoint(max_loc1.x+offsetx, max_loc1.y+offsety), cvPoint(max_loc1.x+offsetx+(temp1->width), max_loc1.y+offsety+(temp1->height)), cvScalar(0, 255, 0 ), 1);


        // Get the middle point of the ball template bottom edge
        int x = max_loc1.x+offsetx+(temp1->width)*0.5+1;
        int y =  max_loc1.y+offsety+(temp1->height)*0.5+1;

        cvCircle(frame, cvPoint(x,y), 3, cvScalar(0, 255, 0 ), CV_FILLED, 8, 0);

        //Display frame
        cvShowImage("Jigsaw puzzles solver", frame);
		resizeWindow("Jigsaw puzzles solver", 600,600);
		moveWindow("Jigsaw puzzles solver",450,100);
        // return the coordinates of each piece.
        int rowLocation =0;
        int colLocation = 0;
        if (x != 0 && y != 0){
            colLocation= x/colUnitWidth+1;
            rowLocation = y/rowUnitHeight+1;
            cout<<colLocation<<" "<<rowLocation<<endl;
			std::ostringstream ss;
			std::string resultStr;
			ss << "The puzzle piece fits at Row number " << rowLocation << " and column number " << colLocation <<std::endl;			
			m_result->setText(QString(ss.str().c_str()));
        }
    }
	
    //Free memory 

    //cvDestroyWindow( "Jigsaw puzzles solver" );

}

void MainDialog::keyPressEvent(QKeyEvent* keyevent)
{
	const QClipboard *clipboard = QApplication::clipboard();
	if(!clipboard)
		return;
	const QMimeData *mimeData = clipboard->mimeData();
	if(!mimeData)
		return;
	if (keyevent->modifiers() == Qt::KeyboardModifier::ControlModifier && keyevent->key()==Qt::Key_S)
    {	
		if (mimeData->hasImage()) {
			m_mainImage->setPixmap(qvariant_cast<QPixmap>(mimeData->imageData()));
			m_selectPieceButton->setEnabled(true);
			m_inputImage = QPixmapToCvMat(qvariant_cast<QPixmap>(mimeData->imageData()));
		}		         
    }else if (keyevent->modifiers() == Qt::KeyboardModifier::ControlModifier && keyevent->key()==Qt::Key_P)
	{
		if (mimeData->hasImage()) {
			m_puzzlePiece->setPixmap(qvariant_cast<QPixmap>(mimeData->imageData()).scaled(QSize(PUZZLEPIECEWIDTH, PUZZLEPIECEHEIGHT)));
			m_detectButton->setEnabled(true);
			m_inputPiece = QPixmapToCvMat(qvariant_cast<QPixmap>(mimeData->imageData()));
		}		         
	}
}