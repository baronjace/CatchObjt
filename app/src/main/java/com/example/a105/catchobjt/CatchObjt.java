package com.example.a105.catchobjt;


import android.content.ContentResolver;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.text.format.DateFormat;
import android.text.format.Time;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.video.Video;

import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;


public class CatchObjt extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";
    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;
    private Mat mRgba,  mRgba2, diff, add, hierarchy;
    boolean diff_str = false;
    boolean tenth = false;
    Timer timer = new Timer(true);
    private int mTime=0, mTime2=0, milliseconds=2000;
    private float touchX;
    private float touchY;
    Point[] pts = new Point[4];
    Rect selection = new Rect(200,250,15,10);
    int thickness = 1;
    private TextView mCde, mSft;
    private Handler mHandler = new Handler();
    private Handler mThreadHandler;
    private HandlerThread mThread;
    Moments moments = new Moments();
    Rect r = new Rect();
    private Calendar mCalendar;
    private SimpleDateFormat df;
    private long startTime;
    private String str;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_catch_objt);
     //   Log.i(TAG, "called onCreate");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_catch_objt);
        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setOnTouchListener(onTouchListener);

        mCde = (TextView) findViewById(R.id.Coordinate);

        operation();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemSwitchCamera = menu.add("Toggle Native/Java camera");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        String toastMesage = new String();
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemSwitchCamera) {
            mOpenCvCameraView.setVisibility(SurfaceView.GONE);
            mIsJavaCamera = !mIsJavaCamera;
            mOpenCvCameraView = (JavaCameraView) findViewById(R.id.tutorial1_activity_java_surface_view);
            toastMesage = "Java Camera";
            mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
            mOpenCvCameraView.setCvCameraViewListener(this);
            mOpenCvCameraView.enableView();
            Toast toast = Toast.makeText(this, toastMesage, Toast.LENGTH_LONG);
            toast.show();
            timer.schedule(new TimerTask1(), 0, milliseconds);
        }
        return true;
    }

    public class TimerTask1 extends TimerTask{
        public void run(){
            mTime ++ ;
            if(mTime == 1){
                diff_str = true;
            }
            if(mTime%1 == 0){
                tenth = true;
            }
        }
    }

    public void onCameraViewStarted(int width ,int height ) {
        mRgba = new Mat(width,height, CvType.CV_8UC1);
        mRgba2 = new Mat(width,height, CvType.CV_8UC1);
        diff = new Mat(width,height, CvType.CV_8UC1);
        add = new Mat(width,height, CvType.CV_8UC1);
        hierarchy = new Mat(width,height, CvType.CV_8UC1);
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(11,11));
        Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(1,1));
        if(tenth == true) {
            mRgba2 = inputFrame.gray().clone();
            Imgproc.threshold(mRgba2, mRgba2, 100, 255, THRESH_BINARY_INV);
            tenth = false;
        }
        mRgba = inputFrame.gray();
        Imgproc.threshold(mRgba, mRgba, 100, 255, THRESH_BINARY_INV);
        Core.add(mRgba, mRgba2, add);
        if(diff_str == true) {
            Core.absdiff(add, mRgba2, diff);
            Imgproc.erode(diff, diff, element);
            Imgproc.dilate(diff, diff, element1);

            List<MatOfPoint> contours = new ArrayList<MatOfPoint>();;
            Core.inRange(diff, new Scalar(255,255,255), new Scalar(255,255,255), diff);
            Imgproc.findContours(diff, contours, hierarchy,Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
            for (int i= 0; i<contours.size(); i++){
                r = Imgproc.boundingRect(contours.get(i));
                int ContourArea = r.height* r.width;
                if(ContourArea > 800){
                    Imgproc.drawContours(diff, contours, i, new Scalar(255,255,255), 3);
                    Imgproc.rectangle(diff, new Point(r.x, r.y), new Point(r.x+r.width, r.y + r.height),
                            new Scalar(255, 255, 255));

                            moments = Imgproc.moments(contours.get(i), false);
                            Point center = new Point(moments.get_m10() / moments.get_m00(),
                                    moments.get_m01() / moments.get_m00());
                            Imgproc.circle(diff, center, 10, new Scalar(255, 255, 255), -1);

                            Imgproc.line(diff, center,
                                    new Point(moments.get_m10() / moments.get_m00(), r.y),
                                    new Scalar(255, 255, 255), thickness);
                            Imgproc.line(diff, center,
                                    new Point(moments.get_m10() / moments.get_m00(), r.y + r.height),
                                    new Scalar(255, 255, 255), thickness);
                            Imgproc.line(diff, center,
                                    new Point(r.x, moments.get_m01() / moments.get_m00()),
                                    new Scalar(255, 255, 255), thickness);
                            Imgproc.line(diff, center,
                                    new Point(r.x + r.width, moments.get_m01() / moments.get_m00()),
                                    new Scalar(255, 255, 255), thickness);

                    new job().execute();
                }
            }

        }
        System.gc();
       if(diff_str == true){
            return diff;
        }
        else{
            return mRgba;
        }
    }

    public void onCameraViewStopped(){
        mRgba2.release();
        mRgba.release();
        diff.release();
    }

    private View.OnTouchListener onTouchListener = new View.OnTouchListener() {
        @Override
        public boolean onTouch(View v, MotionEvent event) {
            touchX = event.getX();
            touchY = event.getY();
            Log.i("touchX", "touchX+");
            Log.i("touchY", "touchY+");


            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:

                    break;
                case MotionEvent.ACTION_MOVE:

                    break;
                case MotionEvent.ACTION_UP:
                    break;
            }
            return true;
        }
    };


    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }

    }
    public class XY{
        public String XV;
        public String YV;
    }

    public class job extends AsyncTask<Void,Void,XY>{
        @Override
        protected XY doInBackground(Void... params) {
            double A =  moments.get_m10()/moments.get_m00();
            double B =  moments.get_m01()/moments.get_m00();
            DecimalFormat df = new DecimalFormat("#.#");
            String X = df.format(A);
            String Y = df.format(B);
            XY xxyy = new XY();
            xxyy.XV = X;
            xxyy.YV = Y;
            return xxyy;
        }

        @Override
        protected void onPostExecute(XY xy) {
            super.onPostExecute(xy);
            mCde.setText(xy.XV + " " + "," + " " + xy.YV);
        }
    }

    public void operation(){
        startTime = System.currentTimeMillis();
        mHandler.removeCallbacks(updateTimer);
        mHandler.postDelayed(updateTimer, 500);
    }

    private Runnable updateTimer = new Runnable() {
        @Override
        public void run() {
            final TextView time = (TextView)findViewById(R.id.sofarTime);
            mHandler.postDelayed(this, 500);
            mCalendar = Calendar.getInstance();
            df = new SimpleDateFormat("HH:mm:ss");
            str = df.format(mCalendar.getTime());
            time.setText(str);
        }
    };

}
