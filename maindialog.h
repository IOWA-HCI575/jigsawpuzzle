#ifndef MAINDIALOG_H
#define MAINDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QKeyEvent>
#include "opencv\cv.h"
#include "opencv\highgui.h"

class MainDialog : public QDialog
{
    Q_OBJECT
private:
    QLabel* m_mainImage;
    QLabel* m_puzzlePiece;
    QPushButton* m_selectPieceButton;
    QPushButton* m_detectButton;
	QString m_wholeImagePath;
	QString m_puzzlePiecePath;

    QLabel* m_result;
	cv::Mat m_inputImage, m_inputPiece;
private slots:
    void uploadImageClicked();
    void uploadPuzzlePieceClicked();
	void process();

public:
    MainDialog(QWidget *parent = 0);
    ~MainDialog();
protected:
	void keyPressEvent(QKeyEvent* keyevent);
};

#endif // MAINDIALOG_H
