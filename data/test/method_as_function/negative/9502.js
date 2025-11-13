// This tests parameters with incorrectly quoted names
const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/emoji.list', {
  params: {
    'foo-bar: true
  }
});