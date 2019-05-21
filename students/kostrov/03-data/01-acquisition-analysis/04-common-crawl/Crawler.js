const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const readline = require('readline');
const S3 = require('aws-sdk/clients/s3');

const config = require('./config.json');

const INDEX = 'crawl-data/CC-MAIN-2019-09/cc-index.paths.gz';

class Crawler {
  constructor(key, s3Config) {
    this.key = key;
    this.s3 = new S3(s3Config);
  }

  getObject(key) {
    return this.s3
      .getObject({
        Bucket: 'commoncrawl',
        Key: key,
      });
  }

  getS3Stream(key) {
    return this.getObject(key).createReadStream();
  }

  getLineStream(key) {
    const s3Stream = this.getS3Stream(key)
      .pipe(zlib.createGunzip());

    return readline.createInterface({
      input: s3Stream,
    });
  }

  async getCdx(key) {
    const lineReader = this.getLineStream(key);
    for await (const line of lineReader) {
      await this.getWarc(line.match(/filename": "([\w-/.]+)/)[1]);
    }
  }

  async getWarc(key) {
    const filename = key.split('/').pop();
    const filepath = path.join('data', filename);
    if (fs.existsSync(filepath)) {
      return;
    }
    await new Promise((resolve, reject) => {
      const output = fs.createWriteStream(filepath);
      this.getObject(key)
        .on('httpData', (chunk) => {
          output.write(chunk);
        })
        .on('error', (err) => {
          console.log(key);
          fs.unlinkSync(filepath);
          reject(err);
        })
        .on('httpDone', () => {
          console.log(`${key} downloaded`);
          output.end();
          resolve();
        })
        .send();
    });
  }

  async crawl() {
    const lineReader = this.getLineStream(this.key);
    for await (const line of lineReader) {
      await this.getCdx(line);
    }
  }
}

(async () => {
  await new Crawler(INDEX, config).crawl();
})();
