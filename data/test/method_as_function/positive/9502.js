// This tests parameters with names that need to be quoted
const axios = require('axios');
axios.get('https://petstore.swagger.io/v2/emoji.list', {
  params: {
    'foo-bar': true
  }
});