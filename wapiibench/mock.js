const axios = require('axios');
const MockAdapter = require('axios-mock-adapter');
const fs = require('fs');

const mock = new MockAdapter(axios);
mock.onAny().reply(config => {
  const url = decodeURIComponent(config.url);
  const queryIndex = url.indexOf('?');
  if (queryIndex === -1) {
    config.url = url;
  } else {
    config.url = url.slice(0, queryIndex);
    const query = url.slice(queryIndex + 1);
    const keyValuePairs = query.split('&');
    if (!Object.hasOwn(config, 'params')) {
      config.params = {};
    }
    for (const pair of keyValuePairs) {
      const valueIndex = pair.indexOf('=');
      if (valueIndex >= 0) {
        const key = pair.slice(0, valueIndex);
        const value = pair.slice(valueIndex + 1);
        if (Object.hasOwn(config.params, key)) {
          console.warn(`Duplicate query parameter '${key}' - overwriting value '${config.params[key]}' with '${value}'`);
        }
        config.params[key] = value;
      } else {
        console.warn(`Could not split keys and values in '${pair}' - assuming value 'true'`);
        config.params[pair] = true;
      }
    }
  }
  if (config.data !== undefined) {
    let data = config.data;
    if (!data || data === 'null') {
      config.data = {};
    } else if (typeof data === 'string' && data.length > 0) {
      if (data.startsWith('{')) {
        config.data = JSON.parse(data);
      } else {
        config.data = {};
        data = decodeURIComponent(data.replaceAll('+', ' '));
        const keyValuePairs = data.split('&');
        for (const pair of keyValuePairs) {
          const valueIndex = pair.indexOf('=');
          let key = pair.slice(0, valueIndex);
          const value = pair.slice(valueIndex + 1);
          if (key.endsWith('[]')) {
            key = key.slice(0, -2);
            if (config.data[key]) {
              config.data[key].push(value);
            } else {
              config.data[key] = [value];
            }
          } else if (key.endsWith(']')) {
            const matches = key.matchAll(/([^[\]]+)/g);
            let tmp = config.data;
            let lastKey;
            let nextKey;
            for (const match of matches) {
              nextKey = match[1];
              if (lastKey) {
                if (!tmp[lastKey]) {
                  tmp[lastKey] = !isNaN(parseInt(nextKey)) ? [] : {};
                }
                tmp = tmp[lastKey];
              }
              lastKey = nextKey;
            }
            tmp[lastKey] = value;
          } else {
            config.data[key] = value;
          }
        }
      }
    } else if (typeof data === 'object' && data.constructor.name === 'FormData') {
      config.data = {};
      const formData = data._streams;
      for (let i = 0; i < formData.length; i += 3) {
        const key = formData[i].match(/name="(.*?)"/)[1];
        const value = formData[i + 1];
        config.data[key] = value;
      }
    } else {
      console.warn(`Unknown type of data '${data}'`);
    }
  }
  fs.writeFile('%s', JSON.stringify(config, null, 2), err => {
    if (err) {
      throw err;
    }
  });
  return [200];
});
