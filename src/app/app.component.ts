import { Component } from '@angular/core';

import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'TF.js simple';

  // ********************************************
  // ******* Built according to tutorial: *******
  // to start the app, follow Angular CLI, node JS tutorial at https://www.youtube.com/watch?v=EBp70tnlFU4
  // TensorFlow.js Quick Start - by Angular Firebase at https://www.youtube.com/watch?v=Y_XM3Bu-4yc&t=224s
  // ********************************************
  
  linearModel: tf.Sequential;
  prediction: any;

  ngOnInit() {
    this.trainNewModel();
  }
  
  async trainNewModel() {
    //Define a model for linear regression
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}));

    //Prepare the model for training: Specify the loss and the optimizer
    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // ********************************************
    // ** Now train the model by feeding it data **
    // ********************************************

    // Training data, completely random stuff
    const xs = tf.tensor1d([0, 1,1.5,2,2.5,3,3.5,4,4.5,5,5.6,6,6.6,7,7.4,8,8.5,9,9.3,10,10.7, -1,-1.5,-2,-2.5,-3,-3.5,-4,-4.5,-5,-5.6,-6,-6.6,-7,-7.4,-8,-8.5,-9,-9.3,-10,-10.7]);
    const ys = tf.tensor1d([0, -1,-1.5,-2,-2.5,-3,-3.5,-4,-4.5,-5,-5.6,-6,-6.6,-7,-7.4,-8,-8.5,-9,-9.3,-10,-10.7, 1,1.5,2,2.5,3,3.5,4,4.5,5,5.6,6,6.6,7,7.4,8,8.5,9,9.3,10,10.7]);

    await this.linearModel.fit(xs, ys);

    console.log('First model trained!');
  }

  linearPrediction(val) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1, 1])) as any;
    
    //Angular donesn't know how to work with tensors, convert to array
    this.prediction = Array.from(output.dataSync())[0];
  }
}
